"""
Inter-agent collision avoidance guide using Monte Carlo score ascent
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class MonteCarloGuide:
    def __init__(self, trajectory_length, state_dim, collision_radius, learning_rate=1e-2):
        self.T = trajectory_length
        self.D = state_dim
        self.radius = collision_radius

        # Guide parameters (defined lazily)
        self.param_loc = None
        self.param_scale = None
        self.optimizer = None

    def initialize_params(self, noisy_traj_batch, alpha_bar_t):
        """
        Initializes the variational parameters to match the input noisy trajectory
        and sets the variance based on the diffusion noise level.
        """
        B, T, D = noisy_traj_batch.shape
        alpha_bar_t = torch.tensor(alpha_bar_t, dtype=torch.float32, device=noisy_traj_batch.device)

        noise_std = torch.sqrt(1.0 - alpha_bar_t).view(1, 1, 1).to(noisy_traj_batch.device)

        self.param_loc = nn.Parameter(noisy_traj_batch.detach().clone().to(noisy_traj_batch.device))
        self.param_scale = noise_std.expand(B, T, D).clone().to(noisy_traj_batch.device)

        self.optimizer = torch.optim.Adam([self.param_loc], lr=1e-2)

    def compute_collision_cost(self, x, noisy_trajs_others):
        """
        Penalizes proximity between agent and others using exponential repulsion.
        x: (B, T, D)
        noisy_trajs_others: (N-1, B, T, D)
        Returns:
            scalar collision penalty
        """
        B, T, D = x.shape
        penalty_total = torch.tensor(0.0, device=x.device)

        for t in range(T):
            for j in range(noisy_trajs_others.shape[0]):
                dist_to_other = torch.norm(x[:, t] - noisy_trajs_others[j, :, t], dim=-1)  # (B,)
                penalty = torch.exp(-10 * (dist_to_other - self.radius))  # (B,)
                penalty_total += penalty.mean()

        return penalty_total

    def compute_log_likelihood(self, x, noisy_traj_batch, alpha_bar_t):
        """
        Computes log p(x_t | x_0) as a Gaussian likelihood.
        """
        mu = torch.sqrt(alpha_bar_t) * x
        sigma = torch.sqrt(1.0 - alpha_bar_t).view(1, 1, 1).expand_as(mu)
        likelihood = Normal(mu, sigma)
        log_prob = likelihood.log_prob(noisy_traj_batch)  # (B, T, D)
        return log_prob.sum(dim=(1, 2)).mean()  # average over batch

    def infer(self, noisy_traj_batch, noisy_trajs_others, alpha_bar_t, factor_steps=50):
        """
        Runs score ascent optimization to enforce collision avoidance.

        Args:
            noisy_traj_batch: (B, T, D) - agent's current noisy trajectory
            noisy_trajs_others: (N-1, B, T, D) - other agents' noisy trajectories
            alpha_bar_t: scalar or (1,) tensor - noise level at timestep t

        Returns:
            Updated trajectory mean: (B, T, D)
        """
        self.initialize_params(noisy_traj_batch, alpha_bar_t)
        for step in range(factor_steps):
            self.optimizer.zero_grad()

            q_dist = Normal(self.param_loc, torch.clamp(self.param_scale, min=1e-4))
            x = q_dist.rsample()  # reparameterized sample from guide

            log_likelihood = self.compute_log_likelihood(x, noisy_traj_batch, alpha_bar_t)
            collision_cost = self.compute_collision_cost(x, noisy_trajs_others)

            loss = -log_likelihood + collision_cost  # minimize -ELBO
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(f"[MC Step {step}] Loss: {loss.item():.4f} | LL: {log_likelihood.item():.4f} | Coll: {collision_cost.item():.4f}")

        return self.param_loc.detach().clone()
