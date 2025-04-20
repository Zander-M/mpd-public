"""
    Factor Graph Guide
"""

import pyro
from pyro.nn import PyroModule, PyroParam
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
import torch

class FactorGraphTrajectoryRefiner(PyroModule):
    def __init__(self, X_noisy, sigma=0.1, lr=1e-3, steps=50):
        super().__init__()
        self.B, self.N, self.H, self.D = X_noisy.shape # batch_size, num_agents, horizon, dim
        self.device = X_noisy.device

        # 

