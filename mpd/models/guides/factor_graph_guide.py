"""
    Inter-agent collision avoidance factor
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class CollisionAvoidanceGuide:
    def __init__(self, trajectory_length, state_dim, collision_radius, learning_rate=1e-8):
        self.T = trajectory_length
        self.D = state_dim
        self.radius = collision_radius

        # Lazy initialization of parameters
        self.param_loc = None
        self.param_scale = None

        self.optimizer = Adam({"lr": learning_rate})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())

    def model(self, noisy_traj_batch, noisy_trajs_others, alpha_bar_t):
        """
        Args:
            noisy_traj_batch: (B, T, D)
            noisy_trajs_others: (N-1, B, T, D)
            alpha_bar_t: scalar or 0-dim tensor
        """
        B, T, D = noisy_traj_batch.shape

        with pyro.plate("batch", B):
            x = pyro.sample("x", dist.Normal(torch.zeros(T, D, device=noisy_traj_batch.device), 
                                             torch.ones(T, D, device=noisy_traj_batch.device)).to_event(2))  # (B, T, D)

            for t in range(T):
                for j in range(noisy_trajs_others.shape[0]):
                    dist_to_other = torch.norm(x[:, t] - noisy_trajs_others[j, :, t], dim=-1)
                    penalty = torch.exp(-10 * (dist_to_other - self.radius))
                    pyro.factor(f"collision_{t}_{j}", -penalty)

            mu = torch.sqrt(alpha_bar_t) * x
            sigma = torch.sqrt(1 - alpha_bar_t)

            pyro.sample("obs", dist.Normal(mu, sigma).to_event(2), obs=noisy_traj_batch)

    def guide(self, noisy_traj_batch, noisy_trajs_others, alpha_bar_t):
        B, T, D = noisy_traj_batch.shape

        if self.param_loc is None or self.param_loc.shape[0] != B:
            # Lazily create parameters per batch
            self.param_loc = pyro.param("traj_loc", torch.zeros(B, T, D))
            self.param_scale = pyro.param("traj_scale", torch.ones(B, T, D) * 0.1,
                                          constraint=dist.constraints.positive)

        with pyro.plate("batch", B):
            pyro.sample("x", dist.Normal(self.param_loc, self.param_scale).to_event(2))

    def infer(self, noisy_traj_agent, noisy_trajs_others, alpha_bar_t, factor_steps=50):
        """
        Args:
            noisy_traj_agent: (B, T, D)
            noisy_trajs_others: (N-1, B, T, D)
        Returns:
            (B, T, D) updated trajectory
        """
        pyro.clear_param_store()

        # Reinitialize variational parameters per call
        B, T, D = noisy_traj_agent.shape
        noise_std = torch.sqrt(1.0-alpha_bar_t).view(1, 1, 1).to(noisy_traj_agent.device)
        pyro.param("traj_loc", noisy_traj_agent.detach().clone())
        pyro.param("traj_scale", noise_std.expand(B, T, D).clone(),
                   constraint=dist.constraints.positive)

        for step in range(factor_steps):
            loss = self.svi.step(noisy_traj_agent, noisy_trajs_others, alpha_bar_t)
            # if step % 10 == 0:
                # print(f"[SVI Step {step}] Loss: {loss:.4f}")
        return self.param_loc.clone().detach()
