"""
    Factor graph with single agent diffusion guidance
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
    def __init__(self, B, H, D, device):
        super().__init__()
        self.B = B
        self.H = H
        self.D = D
        self.device = device

        # Place Holders
        self.loc = PyroParam(torch.zeros(B, H, D, device=device, requires_grad=True))
        self.scales = PyroParam(torch.full((B, H, D), 1.0, device=self.device, requires_grad=True),
                                    constraint=constraints.greater_than(-10))

    def update_guide(self, X_noisy, agent_idx):
        """
            Update Guide with new input values
        """
        # Declare parameters to optimize
        with torch.no_grad():
            self.loc.data.copy_(X_noisy[:, agent_idx])

    def forward(self, *args, **kwargs):

        assert self.loc.requires_grad, "Guide.loc is not differentible!" 

        # Use initial noisy trajectory as sampling guide
        with pyro.plate("batch", self.B):
            pyro.sample("X", dist.Normal(self.loc, self.scales).to_event(2))

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
        self.curr_trajectory = None
        self.starts = None
        self.goals = None
        self.agent_idx = None
        self.alpha_bar = None

        self.collision_threshold = collision_threshold # collision threshold
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

            # We propose a linear interpolation between start and goal as the model
            interp = torch.linspace(0, 1, self.H, device=self.device).view(1, self.H, 1)
            mu = (1 - interp) * self.starts.unsqueeze(1) + interp * self.goals.unsqueeze(1)
            scale = torch.ones_like(mu, device=self.device)
            x = pyro.sample("X", dist.Normal(mu, scale).to_event(2))
            # x = pyro.sample("x", dist.Normal(torch.zeros(self.B, self.H, self.D, device=self.device),
                                            #  torch.ones(self.B, self.H, self.D, device=self.device))
                            # .to_event(2))

            # x = pyro.sample("x", dist.Normal(self.curr_trajectory, 1.0).to_event(2))

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
                    pyro.factor(f"collision_{self.agent_idx}_{j}", penalty)
                else:
                    xj = self.X_noisy[:, j]  # shape: (B, H, D)
                    dists_sq = ((x - xj) ** 2).sum(dim=-1)  # (B, H)

                    # Repulsion & distance mask
                    repulsion = torch.exp(-dists_sq/(2* self.sigma["collision_factor"] ** 2))
                    mask = (dists_sq < self.collision_threshold ** 2).float()

                    # Apply Gaussian repulsion only when agents are too close
                    penalty_terms = repulsion * mask

                    # We inject alpha_bar here. The ealier in the denoising steps, the softer the penalty is.
                    penalty = - self.alpha_bar *penalty_terms.sum(dim=-1) # (B) 
                    pyro.factor(f"collision_{self.agent_idx}_{j}", penalty)
                if not torch.isfinite(penalty).all():
                    raise RuntimeError(f"Invalid penalty at agent {self.agent_idx}, j={j}")

        
# Module 
class FactorGuideSingleAgent(PyroModule):
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

        # Initialize model and guide 
        self.model = FactorModel(B, N, H, D, device, 
                                 collision_threshold=collision_threshold, sigma=sigma)
        self.guide = Guide(B, H, D, device) 
        self.optimizer = SVI(self.model, self.guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())

    def update_inputs(self, X_noisy, agent_idx, alpha_bar):
        """
            Update current model and guidance based on new input
        """
        self.model = FactorModel(self.B, self.N, self.H, self.D, self.device, 
                                 collision_threshold=self.collision_threshold, sigma=self.sigma)
        self.guide = Guide(self.B, self.H, self.D, self.device) 
        self.model.update_model(X_noisy=X_noisy, agent_idx=agent_idx, alpha_bar=alpha_bar)
        self.guide.update_guide(X_noisy=X_noisy, agent_idx=agent_idx)
        self.optimizer = SVI(self.model, self.guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())

    def forward(self, X_noisy, alpha_bar):
        """
            Iterate through agents and return updated trajectories
        """
        X_noisy = X_noisy.detach()
        num_agents = X_noisy.shape[1]
        X_refined = X_noisy.clone().detach()
        alpha_bar = torch.clamp(alpha_bar, 0.0, 1.0).clone().detach()
        # blend = torch.exp(-alpha_bar**2)

        for agent_idx in range(num_agents):
            self.update_inputs(X_noisy=X_noisy, agent_idx=agent_idx, alpha_bar=alpha_bar)
            for step in range(self.steps):
                loss = self.optimizer.step()
                if step % 100 == 0:
                    print(f"Step {step} : loss = {loss}")
            
            refined_paths = self.guide.loc.detach()
            
            print("Delta norm: ", torch.norm(self.guide.loc.detach() - X_noisy[:, agent_idx]))

            # Clipping merge factor, for overflow prevention
            # refined_all[:, agent_idx] = blend * X_noisy[:, agent_idx] + (1 - blend) * refined_paths
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
