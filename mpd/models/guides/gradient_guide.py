import torch
import torch.nn as nn

class FactorGuidedDenoiser(nn.Module):
    def __init__(self, min_dist=0.5, collision_weight=10.0, position_weight=1.0):
        super().__init__()
        self.min_dist = min_dist
        self.collision_weight = collision_weight
        self.position_weight = position_weight

    def forward(self, X, starts, goals, noise_scales):
        """
        X: (N, B, H, D)         - agent trajectories
        starts: (N, B, D)       - agent-specific start locations
        goals: (N, B, D)        - agent-specific goal locations
        noise_scales: float or tensor of shape broadcastable to (N, B)
        """
        N, B, H, D = X.shape
        X = X.clone().detach().requires_grad_(True)

        # Position factor loss
        start = starts[:, :, None, :]         # (N, B, 1, D)
        goal = goals[:, :, None, :]           # (N, B, 1, D)
        loss_start = torch.norm(X[:, :, 0:1] - start, dim=-1)  # (N, B, 1)
        loss_goal  = torch.norm(X[:, :, -1:] - goal, dim=-1)   # (N, B, 1)
        loss_position = self.position_weight * (loss_start + loss_goal).sum()

        # Collision factor
        # Pairwise distance computation
        X1 = X[:, :, None, :, :]   # (N, B, 1, H, D)
        X2 = X[None, :, :, :, :]   # (1, N, B, H, D)
        pairwise_deltas = X1 - X2  # (N, N, B, H, D)
        dists_sq = (pairwise_deltas ** 2).sum(dim=-1)  # (N, N, B, H)

        # Mask out self-collisions
        mask = ~torch.eye(N, dtype=torch.bool, device=X.device)[:, :, None, None]
        dists_sq = dists_sq[mask].view(N, N - 1, B, H)

        # Expand noise_scale to shape (N, B)
        if not torch.is_tensor(noise_scales):
            noise_scales = torch.tensor(noise_scales, dtype=torch.float32, device=X.device)
        sigma_sq = noise_scales ** 2 + 1e-6  # (N, B)
        sigma_sq = sigma_sq.view(N, 1, B, 1)  # broadcastable to (N, N-1, B, H)

        # Compute collision penalty
        penalty = torch.exp(-dists_sq / (2 * sigma_sq))
        penalty = penalty * (dists_sq < self.min_dist ** 2).float()
        loss_collision = self.collision_weight * penalty.sum()

        # Total loss
        total_loss = (loss_position + loss_collision) / (N * B * H)

        # Compute gradients
        grads = torch.autograd.grad(total_loss, X)[0]

        # Update trajectories
        X_updated = X - grads  # basic Euler step

        return X_updated
