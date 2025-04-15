"""
    Factor Graph SVI-based guidance
    Refine all agent trajectories.
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.optim import ClippedAdam

class ReusableRefiner(PyroModule):
    def __init__(self, N, T, D, sigma=0.5, lr=1e-2, steps=100):
        """
        N: number of agents
        T: trajectory horizon
        D: spatial dimension (e.g., 2 for 2D)
        sigma: controls collision penalty width
        lr: learning rate for SVI
        steps: number of optimization steps per call
        """
        super().__init__()
        self.N, self.T, self.D = N, T, D
        self.sigma = sigma
        self.lr = lr
        self.steps = steps
        self.guide = None
        self.svi = None
        self.initialized = False

    def init(self, B):
        """
        Initializes the guide and SVI components for batch size B.
        Only called once unless manually reset.
        """
        class Guide(PyroModule):
            def __init__(self, B, N, T, D):
                super().__init__()
                self.means = PyroParam(torch.randn(B, N, T, D) * 0.1)
                self.log_scales = PyroParam(torch.full((B, N, T, D), -1.0))

            def forward(self, starts, goals):
                with pyro.plate("batch", B):
                    for i in range(N):
                        pyro.sample(f"x_{i}", dist.Normal(self.means[:, i], self.log_scales[:, i].exp()).to_event(2))

        def model(starts, goals):
            interp = torch.linspace(0, 1, self.T).view(1, self.T, 1).to(starts.device)
            with pyro.plate("batch", B):
                trajectories = []
                for i in range(self.N):
                    mu = (1 - interp) * starts[:, i].unsqueeze(1) + interp * goals[:, i].unsqueeze(1)
                    x_i = pyro.sample(f"x_{i}", dist.Normal(mu, 1.0).to_event(2))
                    trajectories.append(x_i)
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        xi, xj = trajectories[i], trajectories[j]
                        dists_sq = ((xi - xj) ** 2).sum(dim=-1)
                        penalty = -dists_sq.sum(dim=-1) / (2 * self.sigma ** 2)
                        pyro.factor(f"collision_{i}_{j}", penalty)

        self.guide = Guide(B, self.N, self.T, self.D)
        self.svi = SVI(model, self.guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())
        self.initialized = True

    def forward(self, X_noisy, alpha_bar, alpha_strength=1.0):
        """
        Refines noisy trajectories based on SVI.
        
        X_noisy: (B, N, T, D) noisy trajectories
        alpha_bar: scalar or tensor in [0, 1], denoising progress (0 = noised, 1 = denoised)
        alpha_strength: multiplier on how strongly to trust refinement

        Returns:
        Refined trajectories of shape (B, N, T, D)
        """
        B = X_noisy.shape[0]
        starts = X_noisy[:, :, 0, :]
        goals = X_noisy[:, :, -1, :]

        if not self.initialized:
            self.init(B)

        for _ in range(self.steps):
            self.svi.step(starts, goals)

        refined = self.guide.means.detach()
        blend = 1.0 - alpha_strength * (1.0 - alpha_bar)
        return blend * X_noisy + (1 - blend) * refined