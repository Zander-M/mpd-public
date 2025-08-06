"""
    Factor graph with centralized guidance
    We encode inter-agent collision constriants and start/goal constraints
"""

import torch
import pyro
import pyro.distributions as dist 
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.optim import ClippedAdam

# Guide 
class Guide(PyroModule):
    """
        Guide for SVI. 
        We use the noisy trajectory of the agent as the guide
    """
    def __init__(self, X_noisy):
        super().__init__()

        self.B, self.N, self.H, self.D = X_noisy.shape
        self.device = X_noisy.device
        self.loc = PyroParam(X_noisy[:, :, 1:-1].clone().detach()) # (B, N, H-2, D)
        self.log_scales = PyroParam(torch.full((self.B, self.N, self.H-2, self.D), -1.0, device=self.device),
                                    constraint=constraints.greater_than(-10))

    def forward(self, *args, **kwargs):
        # Use initial noisy trajectory as sampling guide
        with pyro.plate("agent", self.N), pyro.plate("batch", self.B):
                pyro.sample("z", dist.Normal(self.loc, torch.exp(self.log_scales)).to_event(2)) # (B, N, H-2, D)

# Model

class FactorModel(PyroModule):
    """
        Collision avoidance model, supports trajectory update
        Factors:
            Start-Goal Postion Factor
            Collision constriants avoidance Factor
    """
    def __init__(self, X_noisy, alpha_bar,
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
        self.starts = X_noisy[:, :, 0] # (B, N, D)
        self.goals = X_noisy[:, :, -1] # (B, N, D)
        self.alpha_bar = alpha_bar

        self.collision_threshold = collision_threshold # collision threshold
        self.sigma = sigma # penalty strength for each factor

    def forward(self, *args, **kwargs):
        with pyro.plate("agent", self.N), pyro.plate("batch", self.B):
            # z = pyro.sample("z", dist.Normal(
                # torch.zeros(self.B, self.N, self.H-2, self.D, device=self.device),
                # torch.ones(self.B, self.N, self.H-2, self.D, device=self.device)
            # ).to_event(2))
            # Compute linear interpolation from start to goal (excluding start and goal steps)
            interp = torch.linspace(0, 1, self.H, device=self.device)[1:-1]  # (H-2,)
            interp = interp.view(1, 1, self.H-2, 1)  # (1, 1, H-2, 1) for broadcasting

            # Interpolated mean: (B, N, H-2, D)
            mu = (1 - interp) * self.starts.unsqueeze(2) + interp * self.goals.unsqueeze(2)

            # Standard deviation (you can tune this as needed)
            std = torch.ones_like(mu)

            # Sample latent middle points from prior
            z = pyro.sample("z", dist.Normal(mu, std).to_event(2))  # (B, N, H-2, D)

            # Enforce Start/Goal position Factor
            x = torch.cat([
                self.starts.unsqueeze(2), # (B, N, 1, D)
                z,                        # (B, N, H-2, D)
                self.goals.unsqueeze(2)   # (B, N, 1, D)
            ], dim=2)

            # === Collision Penalty === #
            xi = x.unsqueeze(2) # (B, N, 1, H, D)
            xj = self.X_noisy.unsqueeze(1) # (B, 1, N, H, D)
            dists_sq = ((xi-xj)**2).sum(dim=-1) # (B, N, N, H)

            # Repulsion & distance mask
            repulsion = torch.exp(-dists_sq/(2* self.sigma["collision_factor"] ** 2))
            mask = (dists_sq < self.collision_threshold ** 2).float()

            # Apply Gaussian repulsion only when agents are too close
            eye = torch.eye(self.N, device=self.device).unsqueeze(0).unsqueeze(-1)
            penalty_terms = repulsion * mask * (1-eye)

            # Remove self collision
            penalty = - self.alpha_bar *penalty_terms.sum(dim=[2, 3]) # (B) 
            pyro.factor(f"collision", penalty)

            # === Smoothness Penalty === #
            pos = x[..., :2]  # Only positions, (B, N, H, 2)
            acc = pos[:, :, 2:] - 2 * pos[:, :, 1:-1] + pos[:, :, :-2]  # (B, N, H-2, 2)
            smooth_cost = acc.pow(2).sum(dim=-1)  # (B, N, H-2)
            smoothness_penalty = -smooth_cost.sum(dim=-1) / (2 * self.sigma["smoothness_factor"] ** 2)  # (B,)
            pyro.factor("smoothness", smoothness_penalty)
        
# Module 
class FactorGuideCentralized(PyroModule):
    """
        Factor Guidance Model that refines the trajectory for each agent.
        We condition each agent's trajectory based on other agents' trajectories.
    """
    def __init__(self, B, N, H, D, device, 
                 collision_threshold = 0.2,
                 sigma=
                 {
                    "position_factor": 0.02,
                    "collision_factor": 0.001
                 }, 
                 lr=1e-8, steps=10):

        super().__init__()
        self.B = B # batch size
        self.N = N # num agent
        self.H = H # horizon
        self.D = D # action dim
        self.device = device # data device

        self.collision_threshold = collision_threshold
        self.sigma = sigma
        self.lr = lr
        self.steps = steps

    def instance_setup(self, X_noisy, alpha_bar):
        """
            Update current model and guidance based on new input
        """
        model = FactorModel(X_noisy=X_noisy, alpha_bar=alpha_bar,
                            collision_threshold=self.collision_threshold, sigma=self.sigma)
        guide = Guide(X_noisy=X_noisy)
        optimizer = SVI(model, guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())
        return model, guide, optimizer

    def forward(self, X_noisy, alpha_bar):
        """
            Iterate through agents and return updated trajectories
        """
        alpha_bar = torch.clamp(alpha_bar, 0.0, 1.0)
        blend = 0.5
        # blend = torch.exp(-(1-alpha_bar)**2)

        model, guide, optimizer = self.instance_setup(X_noisy=X_noisy, alpha_bar=alpha_bar)
        for _ in range(self.steps):
            optimizer.step()
            
        refined_paths = guide.loc.clone().detach()
        X_refined = torch.cat([
            model.starts.unsqueeze(2),
            refined_paths,
            model.goals.unsqueeze(2)
        ], dim=2)
        
        # Blend schedule
        X_refined = (blend) * X_noisy + (1-blend) * X_refined
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
