"""
    Adapted from mmd
"""

import os
import pickle
import time

import torch
import yaml
from matplotlib import pyplot as plt

from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

def generate_collision_free_trajectories(
    env_id,
    robot_id,
    num_trajectories_per_context,
    results_dir,
    threshold_start_goal_pos=1.0,
    obstacle_cutoff_margin=0.03,
    n_tries=1000,
    rrt_max_time=300,
    gpmp_opt_iters=500,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
):
    # TODO: add random obstacles in configuration space to increase trajectory diversity

    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context

    # -------------------------------- Hybrid Planner ---------------------------------
    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state_pos=start_state_pos,
        goal_state_pos=goal_state_pos,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
    # sample_based_planner_base = RRTStar(**rrt_connect_params)
    # sample_based_planner = sample_based_planner_base
    sample_based_planner = MultiSampleBasedPlanner(
        sample_based_planner_base,
        n_trajectories=n_trajectories,
        max_processes=-1,
        optimize_sequentially=True
    )

    # Optimization-based planner
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=n_trajectories,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ###############
    # Hybrid planner
    planner = HybridPlanner(
        sample_based_planner,
        opt_based_planner,
        tensor_args=tensor_args
    )

    # Optimize
    trajs_iters = planner.optimize(debug=debug, print_times=True, return_iterations=True)
    trajs_last_iter = trajs_iters[-1]

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter)*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter)*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # save results data dict
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': planner_params['dt'],
        'trajs_iters_coll': trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        'trajs_iters_free': trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )

    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=300)
    plt.close(fig)

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free

def generate_linear_trajectories(
        env_id,
        robot_id,
        num_trajectories_per_context,
        results_dir,
        threshold_start_goal_pos=1.0,
        obstacle_cutoff_margin=0.03,
        n_tries=1000,
        n_support_points=64,
        duration=5.0,
        tensor_args=None,
        is_wait_at_goal=True,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    print("linear planner")
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context
    # -------------------------------- Linear Planner ---------------------------------
    #  The output shape of trajectories is (n_trajectories_for_start_goal_pair, n_support_points, state_dim)
    if is_wait_at_goal:
        # Only allowing velocity v or velocity 0. These are stacked below the positions (vx, vy).
        v_mag = 0.05
    else:
        # Velocity is distance/steps.
        v_mag = torch.linalg.norm(goal_state_pos - start_state_pos) / n_support_points

    if n_trajectories != 1:
        raise ValueError(f"n_trajectories must be 1 for linear planner. Got {n_trajectories}")

    traj_dist = torch.linalg.norm(goal_state_pos - start_state_pos)
    traj_num_points_moving = traj_dist / v_mag
    traj_num_points_moving = torch.floor(traj_num_points_moving).int()
    traj_interpolation_weights = torch.linspace(0, 1, int(traj_num_points_moving), **tensor_args).unsqueeze(1)
    traj = start_state_pos + traj_interpolation_weights * (goal_state_pos - start_state_pos)
    # Add points to the end of the trajectory waiting at the goal, if any exist.
    traj_num_points_waiting = n_support_points - traj_num_points_moving
    if traj_num_points_waiting > 0:
        traj = torch.cat((traj, torch.stack([goal_state_pos] * int(traj_num_points_waiting))))
    # Compute the velocity vectors by finite differencing.
    traj_vel = torch.cat((torch.diff(traj, dim=0), torch.zeros(1, robot.q_dim, **tensor_args)))
    traj = torch.cat((traj.unsqueeze(0), traj_vel.unsqueeze(0)), dim=-1)
    trajs_last_iter = traj

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter) * 100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter) * 100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # Save results data dict.
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': duration / n_support_points,
        'trajs_iters_coll': None,
        'trajs_iters_free': None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=100)
    plt.close(fig)

    # Visualize animated. Uncomment below to safe a GIF.
    # trajs_dense_l = densify_trajs([trajs_last_iter], 1)
    # output_fpath = os.path.join(results_dir, f'robot-traj.gif')
    # file_uri = f'file://{os.path.realpath(output_fpath)}'
    # print(f'Click to open output  directory:file://{os.path.realpath(results_dir)}')
    #
    # print(f'Click to open GIF: {file_uri}')
    # planner_visualizer.animate_multi_robot_trajectories(
    #     trajs_l=trajs_dense_l,
    #     start_state_l=[pos_trajs[i][0] for i in range(len(trajs_dense_l))],
    #     goal_state_l=[pos_trajs[i][-1] for i in range(len(trajs_dense_l))],
    #     plot_trajs=True,
    #     video_filepath=output_fpath,
    #     n_frames=max((2, trajs_dense_l[0].shape[1])),
    #     # n_frames=pos_trajs_iters[-1].shape[1],
    #     anim_time=15.0,
    #     constraints=None,
    #     colors=[plt.cm.tab10(i) for i in range(len(trajs_dense_l))],
    # )

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free



@single_experiment_yaml
def experiment(
    # env_id: str = 'EnvDense2D',
    # env_id: str = 'EnvSimple2D',
    # env_id: str = 'EnvNarrowPassageDense2D',
    # env_id: str = 'EnvSpheres3D',
    env_id: str = 'EnvEmpty2D',

    robot_id: str = 'RobotPointMass',

    n_support_points: int = 64,
    duration: float = 5.0,  # seconds

    # threshold_start_goal_pos: float = 1.0,
    threshold_start_goal_pos: float = 1.83,

    obstacle_cutoff_margin: float = 0.05,

    num_trajectories: int = 1,
    
    trajectory_type: str = "linear", # linear, collsion_free

    # device: str = 'cpu',
    device: str = 'cuda',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = int(time.time()),
    # seed: int = 0,
    # seed: int = 1679258088,
    results_dir: str = "data",

    #######################################
    **kwargs
):
    if debug:
        fix_random_seed(seed)

    print(f'\n\n-------------------- Generating data --------------------')
    print(f'Seed:  {seed}')
    print(f'Env:   {env_id}')
    print(f'Robot: {robot_id}')
    print(f'num_trajectories: {num_trajectories}')

    ####################################################################################################################
    tensor_args = {'device': device, 'dtype': torch.float32}

    metadata = {
        'env_id': env_id,
        'robot_id': robot_id,
        'num_trajectories': num_trajectories
    }
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)

    # Generate trajectories
    if trajectory_type == "linear":
        num_trajectories_coll, num_trajectories_free = generate_linear_trajectories(
            env_id,
            robot_id,
            num_trajectories,
            results_dir,
            threshold_start_goal_pos=threshold_start_goal_pos,
            obstacle_cutoff_margin=obstacle_cutoff_margin,
            n_support_points=n_support_points,
            duration=duration,
            tensor_args=tensor_args,
            is_wait_at_goal=False,
        )
    
    elif trajectory_type == "collision_free":
        num_trajectories_coll, num_trajectories_free = generate_collision_free_trajectories(
            env_id,
            robot_id,
            num_trajectories,
            results_dir,
            threshold_start_goal_pos=threshold_start_goal_pos,
            obstacle_cutoff_margin=obstacle_cutoff_margin,
            n_support_points=n_support_points,
            duration=duration,
            tensor_args=tensor_args,
            debug=debug,
        )

    metadata.update(
        num_trajectories_generated=num_trajectories_coll + num_trajectories_free,
        num_trajectories_generated_coll=num_trajectories_coll,
        num_trajectories_generated_free=num_trajectories_free,
    )
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)


if __name__ == '__main__':
    # Generate n linear trajectories. For san check only
    env_id = "EnvEmpty2D"
    robot_id = "RobotPointMass"
    results_dir = os.path.join("data", f"{env_id}-{robot_id}")
    num_trajectories = 100
    for i in range(num_trajectories):
        run_experiment(
            experiment, 
            args={
                "seed":i, 
                "env_id":env_id, 
                "robot_id":robot_id, 
                "results_dir":results_dir
            }
            )
