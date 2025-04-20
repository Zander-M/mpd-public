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
    def __init__(self, B, N, H, D, device):
        super().__init__()
        self.B = B
        self.N = N
        self.H = H
        self.D = D
        self.device = device

        # Place Holders
        self.loc = PyroParam(torch.zeros(B, N, H, D, device=device))
        self.log_scales = PyroParam(torch.full((B, N, H, D), -1.0, device=self.device),
                                    constraint=constraints.greater_than(-10))

    def update_guide(self, X_noisy):
        """
            Update Guide with new input values
        """
        with torch.no_grad():
            self.loc.copy_(X_noisy.clone().detach())

    def forward(self, *args, **kwargs):

        assert self.loc.shape == (self.B, self.N, self.H, self.D), "Guide.loc shape mismatch"
        assert self.loc.requires_grad, "Guide.loc is not differentible!" 

        # Use initial noisy trajectory as sampling guide
        with pyro.plate("batch", self.B):
            pyro.sample("x", dist.Normal(self.loc, self.log_scales.exp()).to_event(3))

# Model

class FactorModel(PyroModule):
    """
        Collision avoidance model, supports trajectory update
        Factors:
            Start-Goal Postion Factor
            Collision constriants avoidance Factor
    """
    def __init__(self, B, N, H, D, 
                 device,
                 collision_threshold=0.02,
                 sigma={ 
                    "position_factor":0.01, 
                    "collision_factor":0.2 
                    }):
        """
            initializing place holders for input values.
        """
        super().__init__()
        self.B = B # batch size
        self.N = N # num_agents
        self.H = H # horizon
        self.D = D # action dim
        self.device = device
        self.starts = None
        self.goals = None
        self.alpha_bar = None

        self.collision_threshold = collision_threshold # collision threshold
        self.sigma = sigma # penalty strength for each factor

    def update_model(self, X_noisy, alpha_bar):
        """
            Update model and guide based on current input
        """
        self.B, self.N, self.H, self.D = X_noisy.shape
        self.X_noisy = X_noisy
        self.starts = self.X_noisy[:, :, 0] # (B, N, D)
        self.goals = self.X_noisy[:, :, -1] # (B, N, D)
        self.alpha_bar = alpha_bar
    
    def forward(self, *args, **kwargs):
        with pyro.plate("batch", self.B):
            x = pyro.sample("x", dist.Normal(self.X_noisy, 1.0).to_event(3))

            # Start/Goal position Factor
            start_penalty = ((x[:, :, 0] - self.starts) **2).sum(dim=-1)
            goal_penalty = ((x[:, :, -1] - self.goals) **2).sum(dim=-1)
            position_penalty = -(start_penalty + goal_penalty).sum(dim=-1) / (2*self.sigma["position_factor"]**2)
            pyro.factor(f"position", position_penalty)

            # Collision Avoidance Factor
            xi = x.unsqueeze(2) # (B, N, 1, H, D)
            xj = x.unsqueeze(1) # (B, 1, N, H, D)
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

        self.sigma = sigma
        self.lr = lr
        self.steps = steps

        # Initialize model and guide 
        self.model = FactorModel(B, N, H, D, device, 
                                 collision_threshold=collision_threshold, sigma=sigma)
        self.guide = Guide(B, N, H, D, device) 
        self.optimizer = SVI(self.model, self.guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())

    def update_inputs(self, X_noisy, alpha_bar):
        """
            Update current model and guidance based on new input
        """
        self.model.update_model(X_noisy=X_noisy, alpha_bar=alpha_bar)
        self.guide.update_guide(X_noisy=X_noisy)

    def forward(self, X_noisy, alpha_bar):
        """
            Iterate through agents and return updated trajectories
        """
        alpha_bar = torch.clamp(alpha_bar, 0.0, 1.0)
        self.update_inputs(X_noisy=X_noisy, alpha_bar=alpha_bar)

        # blend = torch.exp(-alpha_bar**2)

        for _ in range(self.steps):
            self.optimizer.step()
            
        refined_all = self.guide.loc.detach()

        return refined_all

