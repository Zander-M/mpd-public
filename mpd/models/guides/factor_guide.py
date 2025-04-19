"""
    Factor graph with single agent diffusion guidance
    We encode inter-agent collision constriants and start/goal constraints
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.optim import ClippedAdam

# Guide 

class Guide(PyroModule):
    """
        Guide for SVI. 
        We use the noisy trajectory of the agent as the guide
    """
    def __init__(self, B, H, D, device):
        super().__init__()
        self.B = B
        self.H = H
        self.D = D
        self.device = device

        # Place Holders
        self.loc = PyroParam(torch.zeros(B, H, D, device=device))
        self.log_scales = PyroParam(torch.full((self.B, self.H, self.D), -1.0, device=self.device))


    def update_guide(self, X_noisy, agent_idx):
        """
            Update Guide with new input values
        """
        # Extract current trajectory
        curr_trajectory = X_noisy[:, agent_idx]

        # Declare parameters to optimize
        self.loc.data = curr_trajectory.detach().clone()

    def forward(self, *args, **kwargs):

        # Use initial noisy trajectory as sampling guide
        with pyro.plate("batch", self.B):
            pyro.sample("x", dist.Normal(self.loc, self.log_scales.exp()).to_event(2))

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
                 threshold=0.02,
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
        self.curr_trajectory = None
        self.starts = None
        self.goals = None
        self.agent_idx = None
        self.alpha_bar = None

        self.collision_threshold = threshold # collision threshold
        self.sigma = sigma # penalty strength for each factor

    def update_model(self, X_noisy, agent_idx, alpha_bar):
        """
            Update model and guide based on current input
        """
        self.agent_idx = agent_idx
        self.B, self.N, self.H, self.D = X_noisy.shape
        self.X_noisy = X_noisy
        self.curr_trajectory = X_noisy[:, agent_idx]
        self.starts = self.curr_trajectory[:, 0] # (B, D)
        self.goals = self.curr_trajectory[:, -1] # (B, D)
        self.alpha_bar = alpha_bar
    
    def forward(self, *args, **kwargs):
        with pyro.plate("batch", self.B):
            x = pyro.sample("x", dist.Normal(self.curr_trajectory, 1.0).to_event(2))

            # Start/Goal position Factor
            penalty = -(((x[:, 0]-self.starts)**2).sum(dim=-1) +
                ((x[:, -1]-self.goals)**2).sum(dim=-1)
                ) / (2 * self.sigma["position_factor"] ** 2) # (B)
            pyro.factor(f"position", penalty)

            # Collision Avoidance Factor
            for j in range(self.N):

                # Skip the factor if agent index matches
                if j == self.agent_idx:
                    penalty = torch.zeros((self.B), device=self.device) # (B)
                else:
                    xj = self.X_noisy[:, j]  # shape: (B, T, D)
                    dists_sq = ((x - xj) ** 2).sum(dim=-1)  # (B, T)

                    # Repulsion & distance mask
                    repulsion = torch.exp(-dists_sq/(2* self.sigma["collision_factor"] ** 2))
                    mask = (dists_sq < (self.collision_threshold ** 2)).float()

                    # Apply Gaussian repulsion only when agents are too close
                    penalty_terms = repulsion * mask

                    # We inject alpha_bar here. The ealier in the denoising steps, the softer the penalty is.
                    penalty = - self.alpha_bar *penalty_terms.sum(dim=-1) # (B) 

                pyro.factor(f"collision_{self.agent_idx}_{j}", penalty)

        
# Module 
class FactorGuide(PyroModule):
    """
        Factor Guidance Model that refines the trajectory for each agent.
        We condition each agent's trajectory based on other agents' trajectories.
    """
    def __init__(self, B, N, H, D, device, 
                 collision_threshold = 0.02,
                 sigma=
                 {
                    "position_factor": 0.02,
                    "collision_factor": 0.01
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
        self.model = FactorModel(B, N, H, D, device, threshold=collision_threshold, sigma=sigma)
        self.guide = Guide(B, H, D, device) 
        self.optimizer = SVI(self.model, self.guide, ClippedAdam({"lr": self.lr}), loss=JitTrace_ELBO())

    def update_inputs(self, X_noisy, agent_idx, alpha_bar):
        """
            Update current model and guidance based on new input
        """
        self.model.update_model(X_noisy=X_noisy, agent_idx=agent_idx, alpha_bar=alpha_bar)
        self.guide.update_guide(X_noisy=X_noisy, agent_idx=agent_idx)

    def forward(self, X_noisy, alpha_bar):
        """
            Iterate through agents and return updated trajectories
        """
        num_agents = X_noisy.shape[1]
        refined_all = X_noisy.clone()
        alpha_bar = torch.clamp(alpha_bar, 0.0, 1.0)
        blend = torch.exp(torch.tensor(-alpha_bar**2, device=self.device))

        for agent_idx in range(num_agents):
            self.update_inputs(X_noisy, agent_idx, alpha_bar)
            for _ in range(self.steps):
                self.optimizer.step()
            
            refined_paths = self.guide.loc.detach()

            # Clipping merge factor, for overflow prevention
            refined_all[:, agent_idx] = blend * X_noisy[:, agent_idx] + (1 - blend) * refined_paths
        return refined_all

