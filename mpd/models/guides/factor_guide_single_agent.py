"""
    Factor graph with single agent diffusion guidance
    We encode inter-agent collision constriants and start/goal constraints
"""

import torch
from torch import nn
import pyro
import pyro.distributions as dist 
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.optim import ClippedAdam
from pyro import poutine
from pyro.poutine import trace # debug

# Guide 
class Guide(PyroModule):
    """
        Guide for SVI. 
        We use the noisy trajectory of the agent as the guide
    """
    def __init__(self, X_noisy, agent_idx):
        super().__init__()

        self.B, _, self.H, self.D = X_noisy.shape
        self.device = X_noisy.device
        self.agent_idx = agent_idx
        self.loc = PyroParam(X_noisy[:, agent_idx, 1:-1].clone().detach()) # (B, H-2, D)
        self.log_scales = PyroParam(torch.full((self.B, self.H-2, self.D), -1.0, device=self.device),
                                    constraint=constraints.greater_than(-10))

    def forward(self, *args, **kwargs):
        # Use initial noisy trajectory as sampling guide
        with pyro.plate("batch", self.B):
            pyro.sample(f"z", dist.Normal(self.loc, torch.exp(self.log_scales)).to_event(2))

# Model
class FactorModel(PyroModule):
    """
        Collision avoidance model, supports trajectory update
        Factors:
            Start-Goal Postion Factor
            Collision constriants avoidance Factor
    """
    def __init__(self, X_noisy, agent_idx, alpha_bar,
                 collision_threshold=0.02,
                 sigma={ 
                    "position_factor":0.01, 
                    "collision_factor":0.2 
                    }):
        """
            initializing place holders for input values.
        """
        super().__init__()
        self.X_noisy = X_noisy
        self.B, self.N, self.H, self.D = X_noisy.shape
        self.device = X_noisy.device
        self.curr_trajectory = X_noisy[:, agent_idx]
        self.agent_idx = agent_idx
        self.starts = self.curr_trajectory[:, 0] # (B, D)
        self.goals = self.curr_trajectory[:, -1] # (B, D)
        self.alpha_bar = alpha_bar

        self.collision_threshold = collision_threshold # collision threshold
        self.sigma = sigma # penalty strength for each factor

    def forward(self, *args, **kwargs):
        with pyro.plate("batch", self.B):

            # Force the start and goal position
            z = pyro.sample("z", dist.Normal(
                torch.zeros(self.B, self.H-2, self.D, device=self.device),
                torch.ones(self.B,self.H-2, self.D, device=self.device)
            ).to_event(2))

            x = torch.cat([
                self.starts.unsqueeze(1),  # (B, 1, D)
                z,                         # (B, H-2, D)
                self.goals.unsqueeze(1)    # (B, 1, D)
            ], dim=1)  # (B, H, D)

            # === Collision Avoidance Factor (harsh) === #
            xi = x[..., :2].unsqueeze(1)         # (B, 1, H, 2)
            xj = self.X_noisy[..., :2]           # (B, N, H, 2)
            dists_sq = ((xi - xj) ** 2).sum(dim=-1)  # (B, N, H)

            # Harsher repulsion: inverse distance and exponential
            epsilon = 1e-6
            repulsion = torch.exp(-dists_sq / (2 * self.sigma["collision_factor"] ** 2)) / (dists_sq + epsilon)

            mask = (dists_sq < self.collision_threshold ** 2).float()
            penalty_terms = repulsion * mask  # (B, N, H)

            # Mask self-collision
            agent_mask = torch.ones(self.N, device=self.device)
            agent_mask[self.agent_idx] = 0.0
            penalty_terms *= agent_mask.view(1, self.N, 1)  # (B, N, H)

            penalty = -self.alpha_bar * penalty_terms.sum(dim=[1, 2])  # (B,)
            pyro.factor("collision", penalty)

            # === Smoothness Penalty === #
            pos = x[..., :2]  # Only positions, (B, H, 2)
            acc = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]  # (B, H-2, 2)
            smooth_cost = acc.pow(2).sum(dim=-1)  # (B, H-2)
            smoothness_penalty = -smooth_cost.sum(dim=-1) / (2 * self.sigma["smoothness_factor"] ** 2)  # (B,)
            pyro.factor("smoothness", smoothness_penalty)

        
# Module 
class FactorGuideSingleAgent:
    """
        Factor Guidance Model that refines the trajectory for each agent.
        We condition each agent's trajectory based on other agents' trajectories.
    """
    def __init__(self, B, N, H, D, device, 
                 collision_threshold = 0.2 ,
                 sigma=
                 {
                    "position_factor": 0.02,
                    "collision_factor": 0.0001
                 }, 
                 lr=1e-8, steps=10):

        self.B = B # batch size
        self.N = N # num agent
        self.H = H # horizon
        self.D = D # action dim
        self.device = device # data device
        self.collision_threshold = collision_threshold
        self.sigma = sigma
        self.lr = lr
        self.steps = steps

    def instance_setup(self, X_noisy, agent_idx, alpha_bar):
        """
            Setup an SVI instance based on inputs.
        """
        model = FactorModel(X_noisy=X_noisy, agent_idx=agent_idx, alpha_bar=alpha_bar,
                                 collision_threshold=self.collision_threshold, sigma=self.sigma)
        guide = Guide(X_noisy=X_noisy, agent_idx=agent_idx) 
        optimizer = SVI(model, guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())
        return model, guide, optimizer

    def forward(self, X_noisy, alpha_bar):
        """
            Iterate through agents and return updated trajectories
        """
        num_agents = X_noisy.shape[1]
        X_refined = torch.zeros_like(X_noisy, device=self.device)
        # blend = 1.0 - torch.exp(-30 * (1.0 - alpha_bar)) 
        blend = torch.exp(-(1-alpha_bar)**2) 

        for agent_idx in range(num_agents):
            pyro.clear_param_store()
            model, guide, optimizer = self.instance_setup(X_noisy=X_noisy, agent_idx=agent_idx, alpha_bar=alpha_bar)
            for step in range(self.steps):
                loss = optimizer.step()
                # if step % 500 == 0:
                #     print(f"Step {step} loss:", loss)
            refined_paths = guide.loc.clone().detach()
            refined_paths = torch.cat([
                model.starts.unsqueeze(1),
                refined_paths,
                model.goals.unsqueeze(1)
            ], dim=1)
            # X_refined[:, agent_idx] = (blend)* X_noisy[:, agent_idx] + (1-blend) * refined_paths
            X_refined[:, agent_idx] = refined_paths
        return X_refined

    ## Utils
    def compute_refinement(self, X_noisy, X_refined):
        """
            Show how much the factor model changed the noisy trajectory
        """
        delta = X_refined - X_noisy  # (B, N, H, D)
        l2 = delta.pow(2).sum(dim=-1).sqrt()  # (B, N, H)
        mean_l2_per_agent = l2.mean(dim=[0, 2]).cpu().numpy()  # (N,)

        start_change = (X_refined[:, :, 0] - X_noisy[:, :, 0]).norm(dim=-1).mean(dim=0)  # (N,)
        goal_change = (X_refined[:, :, -1] - X_noisy[:, :, -1]).norm(dim=-1).mean(dim=0)  # (N,)

        return {
            "mean_l2_per_agent": mean_l2_per_agent,
            "mean_start_change": start_change.cpu().numpy(),
            "mean_goal_change": goal_change.cpu().numpy(),
        }

if __name__ == "__main__":
    # simple visual check to make sure the model is ok
    from pyro import poutine
    from pyro.contrib.autoname import scope
    from pyro.contrib.autoname import name_count  # for unique names
    from pyro import render_model

    device = "cpu"
    B = 10
    N = 2
    H = 64
    D = 4
    # Example: dummy inputs
    x_noisy = torch.randn(B, N, H, D, device=device)
    alpha_bar = torch.tensor(0.5, device=device)

    model = FactorModel(B, H, H, D, device)
    guide = Guide(B, H, D, device=device)

    # Update model and guide with these inputs
    model.update_model(x_noisy, agent_idx=0, alpha_bar=alpha_bar)
    guide.update_guide(x_noisy, agent_idx=0)

    # Wrap model and guide calls in `poutine.trace`
    guide_trace = poutine.trace(guide).get_trace()
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace()

    # Render the graphical model
    render_model(
        model=model,
        model_args=(),
        model_kwargs={},
        filename="model_graph.svg"
    )