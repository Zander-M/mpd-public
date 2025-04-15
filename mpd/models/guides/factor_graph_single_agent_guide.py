"""
    Factor Graph SVI-based guidance
    Refine one agent's trajectory based on other agents trajectoies
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.optim import ClippedAdam


class PerAgentRefiner(PyroModule):
    def __init__(self, N, T, D, sigma=0.5, lr=1e-2, steps=100):
        super().__init__()
        self.N, self.T, self.D = N, T, D
        self.sigma = sigma
        self.lr = lr
        self.steps = steps
        self.optimizers = {}  # per-agent SVI objects
        self.guides = {}      # per-agent guides
        self.initialized = {}

    def init(self, B, agent_idx):
        class Guide(PyroModule):
            def __init__(self, B, T, D):
                super().__init__()
                self.means = PyroParam(torch.randn(B, T, D) * 0.1)
                self.log_scales = PyroParam(torch.full((B, T, D), -1.0))

            def forward(self, starts, goals, others_fixed):
                with pyro.plate("batch", B):
                    pyro.sample("x", dist.Normal(self.means, self.log_scales.exp()).to_event(2))

        def make_model(agent_idx):
            def model(starts, goals, others_fixed):
                interp = torch.linspace(0, 1, self.T).view(1, self.T, 1).to(starts.device)
                with pyro.plate("batch", B):
                    mu = (1 - interp) * starts[:, agent_idx].unsqueeze(1) + interp * goals[:, agent_idx].unsqueeze(1)
                    x = pyro.sample("x", dist.Normal(mu, 1.0).to_event(2))
                    for j in range(self.N):
                        if j == agent_idx:
                            continue
                        xj = others_fixed[:, j]  # shape: (B, T, D)
                        dists_sq = ((x - xj) ** 2).sum(dim=-1)
                        penalty = -dists_sq.sum(dim=-1) / (2 * self.sigma ** 2)
                        pyro.factor(f"collision_{agent_idx}_{j}", penalty)
            return model

        guide = Guide(B, self.T, self.D)
        model = make_model(agent_idx)
        svi = SVI(model, guide, ClippedAdam({"lr": self.lr}), loss=Trace_ELBO())

        self.guides[agent_idx] = guide
        self.optimizers[agent_idx] = svi
        self.initialized[agent_idx] = True

    def forward(self, X_noisy, alpha_bar, agent_idx, alpha_strength=1.0):
        """
        X_noisy: (B, N, T, D) noisy trajectories
        alpha_bar: scalar or tensor in [0, 1]
        agent_idx: integer in [0, N-1] — the agent to refine
        """
        B = X_noisy.shape[0]
        starts = X_noisy[:, :, 0, :]
        goals = X_noisy[:, :, -1, :]
        others_fixed = X_noisy.detach()  # treat others as fixed

        if agent_idx not in self.initialized:
            self.init(B, agent_idx)

        svi = self.optimizers[agent_idx]
        guide = self.guides[agent_idx]

        for _ in range(self.steps):
            svi.step(starts, goals, others_fixed)

        refined_agent = guide.means.detach()  # (B, T, D)
        refined_all = X_noisy.clone()
        blend = 1.0 - alpha_strength * (1.0 - alpha_bar)
        refined_all[:, agent_idx] = blend * X_noisy[:, agent_idx] + (1 - blend) * refined_agent

        return refined_all
