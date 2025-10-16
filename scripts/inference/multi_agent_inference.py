"""
    Multi-agent inference
"""


import os
import pickle
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, get_type_hints

import einops
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn

from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer


# Guides
from mpd.models.guides import FactorGuideSingleAgent, FactorGuideCentralized

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../data_trained_models/'

@single_experiment_yaml
def experiment(
    ########################################################################################################################
    # Experiment configuration
    # model_id: str = 'EnvDense2D-RobotPointMassFactor',
    # model_id: str = 'EnvNarrowPassageDense2D-RobotPointMassFactor',
    # model_id: str = 'EnvSimple2D-RobotPointMassFactor',
    model_id: str = 'EnvEmpty2D-RobotPointMass',
    # planner_alg: str = 'diffusion_prior',
    # planner_alg: str = 'diffusion_prior_then_guide',
    planner_alg: str = 'mpd',
    use_guide_on_extra_objects_only: bool = False,
    # num_agents: int = 1, # sanity check
    num_agents: int = 16, # number of agents
    n_samples: int = 128,
    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 5,
    n_diffusion_steps_without_noise: int = 0,

    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,

    factor_num_interpolated_points_for_collision: float = 1.5,

    trajectory_duration: float = 5.0,  # currently fixed
    ########################################################################
    # Guidance Related
    # guidance_model: str = "FactorGuideSingleAgent",
    guidance_model: str = "FactorGuideCentralized",
    # guidance_model: str = "None",

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    render: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',

    ########################################################################
    **kwargs
):
    ########################################################################################################################
    fix_random_seed(seed)

    torch_device = get_torch_device(device)
    tensor_args = {'device': torch_device, 'dtype': torch.float32}

    ########################################################################################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')
    run_prior_only = False
    run_prior_then_guidance = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_guide':
        run_prior_then_guidance = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
    else:
        raise NotImplementedError

    ########################################################################################################################
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, 'results_inference', str(seed))
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    ########################################################################################################################
    # Load dataset with env, robot, task
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=False,
        obstacle_cutoff_margin=0.05,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points # type:ignore
    env = dataset.env # type:ignore
    robot = dataset.robot # type:ignore
    task = dataset.task # type:ignore

    dt = trajectory_duration / n_support_points  # time interval for finite differences

    # set robot's dt
    robot.dt = dt

    ########################################################################################################################
    # Load prior model
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim, # type:ignore
        n_support_points=dataset.n_support_points,  # type: ignore
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    
    # create a shared model for multiple agents. We denoise use the same model
    # under different constraints and 

    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
        map_location=tensor_args['device'])
    )
    diffusion_model.eval()
    model = diffusion_model

    freeze_torch_model_params(model)
    model = torch.compile(model)
    model.warmup(horizon=n_support_points, device=torch_device) 

    ########################################################################################################################
    start_goal_pairs = []
    start_goal_type = "CIRCULAR" # "CIRCULAR" or "RANDOM"

    # Random start and final positions for each agent
    if start_goal_type == "CIRCULAR":
            # Circular initialization for testing in empty space
            radius = 0.8
            start_rad = torch.arange(0, 2*torch.pi, 2*torch.pi/num_agents)
            goal_rad = start_rad + torch.pi
            starts = torch.stack([torch.cos(start_rad)*radius, torch.sin(start_rad)*radius]).T
            goals = torch.stack([torch.cos(goal_rad)*radius, torch.sin(goal_rad)*radius]).T
            start_goal_pairs = [(start, goal) for (start, goal) in zip(starts, goals)]
            start_goal_tensor = torch.stack(
                [torch.stack([start, goal]) for (start, goal) in start_goal_pairs]
            ).to(device)
    else:
        # Default behavior is random collision free start and goal 
        for agent_rad in range(num_agents):
            n_tries = 100
            start, goal= None, None
            for _ in range(n_tries):
                q_free = task.random_coll_free_q(n_samples=2)
                start = q_free[0]
                goal = q_free[1]

                if torch.linalg.norm(start - goal) > dataset.threshold_start_goal_pos: # type: ignore
                    start_goal_pairs.append((start, goal))
                    break

            if start is None or goal is None:
                raise ValueError(f"Agent {agent_rad} failed to find colilsion free start/goal")
        
        start_goal_tensor = torch.stack(
            [torch.stack([start, goal]) for (start, goal) in start_goal_pairs]
        )

    ########################################################################################################################
    # Run motion planning inference

    ########
    # normalize start and goal positions

    hard_conds_all = [
        dataset.get_hard_conditions(start_goal_tensor[i], normalize=True) # type: ignore
        for i in range(num_agents)
    ]
    context = None

    ########
    # Set up the planning costs

    # Cost collisions
    cost_collision_l = []
    weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
    if use_guide_on_extra_objects_only:
        collision_fields = task.get_collision_fields_extra_objects()
    else:
        collision_fields = task.get_collision_fields()

    for collision_field in collision_fields:
        cost_collision_l.append(
            CostCollision(
                robot, n_support_points,
                field=collision_field,
                sigma_coll=1.0,
                tensor_args=tensor_args
            )
        )
        weights_grad_cost_l.append(weight_grad_cost_collision)

    # Cost smoothness
    cost_smoothness_l = [
        CostGPTrajectory(
            robot, n_support_points, dt, sigma_gp=1.0,
            tensor_args=tensor_args
        )
    ]
    weights_grad_cost_l.append(weight_grad_cost_smoothness)

    ####### Cost composition
    cost_func_list = [
        *cost_collision_l,
        *cost_smoothness_l
    ]

    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list,
        weights_cost_l=weights_grad_cost_l,
        tensor_args=tensor_args
    )

    ########
    # Guiding manager
    guide = GuideManagerTrajectoriesWithVelocity(
            dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=tensor_args,
    )

    t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps) # type: ignore
    sample_fn_kwargs = dict(
        guide = None, # No guide
        # guide=None if run_prior_then_guidance or run_prior_only else guide,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,
    )

    ######## Initial Factor Guide
    # TODO: implement factor graph based guide here
    if guidance_model == "FactorGuideSingleAgent":
        print("Using Factor Guide Single Agent")

        # Guide params
        model_guide_param = dict(
            B=n_samples,
            N=num_agents,
            H=n_support_points,
            D=dataset.state_dim, # type: ignore
            device=device,
            collision_threshold = 0.5,
            sigma = {
            # Smaller -> Stronger penalty
               "position_factor": 1e4,
               "collision_factor": 1e-3,
               "smoothness_factor": 1e-3
            },
            lr = 1e-2,
            steps = 100,
        )
        model_guide = FactorGuideSingleAgent

    elif guidance_model == "FactorGuideCentralized":
        print("Using Factor Guide Centralized")

        # Guide params
        model_guide_param = dict(
            B=n_samples,
            N=num_agents,
            H=n_support_points,
            D=dataset.state_dim, # type: ignore
            device=device,
            collision_threshold = 0.8,
            sigma = {
            # Denominator. Smaller -> Stronger penalty
               "position_factor": 1e4,
               "collision_factor": 1e-7,
               "smoothness_factor": 5e-3
            },
            lr = 1e-2,
            steps = 500,
        )
        
        model_guide = FactorGuideCentralized   

    elif guidance_model == "MonteCarlo":
        pass
        # model_guide = MonteCarloGuide(
            # n_support_points, dataset.state_dim, 1e-8
        # ) 
        # model_step = 1 
    else:
        print("No guidance model used.")
        model_guide = None
        model_guide_param = None
    ########
    # Sample trajectories with the diffusion/cvae model

    # Initialize single agent trajectories, iteratively denoise each agent's trajectory
    # batch size (n_samples) is used here
    # (B, N, H, D)
    trajs_normalized = torch.randn(n_samples, num_agents, n_support_points, dataset.state_dim, **tensor_args) # type: ignore

    trajs_normalized_iters = []

    with TimerCUDA() as timer_model_sampling:
        for t in reversed(range(-n_diffusion_steps_without_noise, model.n_diffusion_steps)): # type: ignore
            trajs_normalized = model.run_inference_one_step( # type: ignore
                trajs_normalized=trajs_normalized,
                t=t,
                context=None,
                hard_conds=hard_conds_all,
                horizon=n_support_points,
                sample_fn=ddpm_sample_fn,
                model_guide=model_guide, # type: ignore
                model_guide_param=model_guide_param, # type: ignore
                debug=debug,
                **sample_fn_kwargs,
            )
            trajs_normalized_iters.append(trajs_normalized.clone())

    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed

    # Stack over time to get (T, B, N, H, D)
    trajs_iters = torch.stack(
        [dataset.unnormalize_trajectories(t) for t in trajs_normalized_iters], dim=0 # type: ignore
    )

    # Get positions: (T, B, N, H, 2)
    trajs_iters_all_agents_pos = robot.get_position(trajs_iters)

    ########
    # run extra guiding steps without diffusion
    if run_prior_then_guidance:
        n_post_diffusion_guide_steps = (t_start_guide + n_diffusion_steps_without_noise) * n_guide_steps
        with TimerCUDA() as timer_post_model_sample_guide:
            trajs = trajs_normalized_iters[-1]
            trajs_post_diff_l = []
            for i in range(n_post_diffusion_guide_steps):
                trajs = guide_gradient_steps(
                    trajs,
                    hard_conds=hard_conds_all,
                    guide=guide,
                    n_guide_steps=1,
                    unnormalize_data=False
                )
                trajs_post_diff_l.append(trajs)

            chain = torch.stack(trajs_post_diff_l, dim=1)
            chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
            trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
        print(f't_post_diffusion_guide: {timer_post_model_sample_guide.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed + timer_post_model_sample_guide.elapsed

    # unnormalize trajectory samples from the models
    multi_agent_stats = {}
    if num_agents > 1:
        pos_final = trajs_iters_all_agents_pos[-1]  # (B, N, H, D_pos)
        pairwise_offsets = pos_final[:, :, None, :, :] - pos_final[:, None, :, :, :]
        pairwise_distances = torch.linalg.norm(pairwise_offsets, dim=-1)  # (B, N, N, H)

        diag_mask = torch.eye(num_agents, dtype=torch.bool, device=pairwise_distances.device).view(1, num_agents, num_agents, 1)
        pairwise_distances_excl_self = pairwise_distances.masked_fill(diag_mask, float('inf'))

        min_pairwise_dist_per_agent = pairwise_distances_excl_self.min(dim=-1).values.min(dim=-1).values  # (B, N)
        min_pairwise_dist_per_traj = min_pairwise_dist_per_agent.min(dim=-1).values  # (B,)

        valid_mask = (~diag_mask).expand_as(pairwise_distances)
        valid_counts = valid_mask.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        mean_pairwise_distance = (pairwise_distances * valid_mask.float()).sum(dim=(1, 2, 3)) / valid_counts

        collision_threshold = 0.5
        if isinstance(model_guide_param, dict) and 'collision_threshold' in model_guide_param:
            collision_threshold = float(model_guide_param['collision_threshold'])

        pairwise_violations = (pairwise_distances < collision_threshold) & valid_mask
        violation_fraction = pairwise_violations.float().sum(dim=(1, 2, 3)) / valid_counts

        print(f'multi-agent min pairwise distance (mean over batch): {min_pairwise_dist_per_traj.mean().item():.4f}')
        print(f'multi-agent pairwise violation fraction (mean over batch): {violation_fraction.mean().item():.4f}')

        multi_agent_stats = {
            'pairwise_distance_min_per_agent': min_pairwise_dist_per_agent.detach().cpu(),
            'pairwise_distance_min_per_traj': min_pairwise_dist_per_traj.detach().cpu(),
            'pairwise_distance_mean_per_traj': mean_pairwise_distance.detach().cpu(),
            'pairwise_violation_fraction_per_traj': violation_fraction.detach().cpu(),
            'pairwise_violation_threshold': collision_threshold,
        }

    else: 

        trajs_final = trajs_iters[-1]

        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(trajs_final, return_indices=True)

        ########################################################################################################################
        # Compute motion planning metrics
        print(f'\n----------------METRICS----------------')
        print(f't_total: {t_total:.3f} sec')

        success_free_trajs = task.compute_success_free_trajs(trajs_final)
        fraction_free_trajs = task.compute_fraction_free_trajs(trajs_final)
        collision_intensity_trajs = task.compute_collision_intensity_trajs(trajs_final)

        print(f'success: {success_free_trajs}')
        print(f'percentage free trajs: {fraction_free_trajs*100:.2f}')
        print(f'percentage collision intensity: {collision_intensity_trajs*100:.2f}')

        # compute costs only on collision-free trajectories
        traj_final_free_best = None
        idx_best_traj = None
        cost_best_free_traj = None
        cost_smoothness = None
        cost_path_length = None
        cost_all = None
        variance_waypoint_trajs_final_free = None
        if trajs_final_free is not None:
            cost_smoothness = compute_smoothness(trajs_final_free, robot)
            print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

            cost_path_length = compute_path_length(trajs_final_free, robot)
            print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

            # compute best trajectory
            cost_all = cost_path_length + cost_smoothness
            idx_best_traj = torch.argmin(cost_all).item()
            traj_final_free_best = trajs_final_free[idx_best_traj]
            cost_best_free_traj = torch.min(cost_all).item()
            print(f'cost best: {cost_best_free_traj:.3f}')

            # variance of waypoints
            variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, robot)
            print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

        print(f'\n--------------------------------------\n')

        ########################################################################################################################
        # Save data
        results_data_dict = {
            'trajs_iters': trajs_iters,
            'trajs_final_coll': trajs_final_coll,
            'trajs_final_coll_idxs': trajs_final_coll_idxs,
            'trajs_final_free': trajs_final_free,
            'trajs_final_free_idxs': trajs_final_free_idxs,
            'success_free_trajs': success_free_trajs,
            'fraction_free_trajs': fraction_free_trajs,
            'collision_intensity_trajs': collision_intensity_trajs,
            'idx_best_traj': idx_best_traj,
            'traj_final_free_best': traj_final_free_best,
            'cost_best_free_traj': cost_best_free_traj,
            'cost_path_length_trajs_final_free': cost_smoothness,
            'cost_smoothness_trajs_final_free': cost_path_length,
            'cost_all_trajs_final_free': cost_all,
            'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free,
            't_total': t_total,
            'multi_agent_stats': multi_agent_stats,
        }
        with open(os.path.join(results_dir, 'results_data_dict.pickle'), 'wb') as handle:
            pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################
    # Render results
    if render:
        trajs_iters_all_agents_pos = trajs_iters_all_agents_pos.permute(1, 0, 2, 3, 4) # (B, T, N, H, D)
        planner_visualizer = PlanningVisualizer(
            task=task
        )
        fig, axs = planner_visualizer.render_multi_robot_trajectories_steps(start_goal_pairs=start_goal_pairs, trajs=trajs_iters_all_agents_pos)
        plt.savefig("final_trajectory.png")
        plt.close(fig)
        planner_visualizer.animate_opt_iters_multi_robots(start_goal_pairs=start_goal_pairs, trajs=trajs_iters_all_agents_pos)
        # TODO: fill in here
        planner_visualizer.animate_multi_robot_trajectories(start_goal_pairs=start_goal_pairs, trajs=trajs_iters_all_agents_pos, n_interpolation=1)
    return 


if __name__ == '__main__':
    # Ensure the launcher sees concrete annotation objects despite postponed evaluation.
    experiment.__annotations__ = get_type_hints(experiment)
    # Leave unchanged
    run_experiment(experiment)
