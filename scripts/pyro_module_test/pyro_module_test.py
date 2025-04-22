"""
    Factor graph with single agent diffusion guidance
    We encode inter-agent collision constriants and start/goal constraints
"""

import torch
from torch import nn
import pyro
import pyro.distributions as dist 
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroParam
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
from pyro import poutine
from pyro.poutine import trace # debug

# Guide 

class Guide(PyroModule):
    """
        Guide for SVI. 
        We use the noisy trajectory of the agent as the guide
    """
    def __init__(self, X_noisy):
        super().__init__()

        self.B, self.H = X_noisy.shape
        self.device = X_noisy.device

        # Place Holders
        self.loc = PyroParam(torch.zeros(self.B, self.H, device=self.device))
        self.log_scales = PyroParam(torch.full((self.B, self.H), -1.0, device=self.device),
                                    constraint=constraints.greater_than(-10))

    def forward(self, *args, **kwargs):

        assert self.loc.requires_grad, "Guide.loc is not differentible!" 

        # Use initial noisy trajectory as sampling guide
        with pyro.plate("batch", self.B):
            pyro.sample("x", dist.Normal(self.loc, torch.exp(self.log_scales)).to_event(1))

# Model

class FactorModel(PyroModule):

    def __init__(self, X_noisy):
        """
            initializing place holders for input values.
        """
        super().__init__()
        self.X_noisy = X_noisy
        self.B, self.H = X_noisy.shape
        self.device = X_noisy.device

    def forward(self, *args, **kwargs):
        with pyro.plate("batch", self.B):
            x = pyro.sample(
                "x", 
                dist.Normal(torch.zeros(self.B, self.H, device=self.device),
                            torch.ones(self.B, self.H, device=self.device)
                            ).to_event(1)
            )
            y = torch.ones(self.B, self.H, device=self.device)
            penalty = ((x - y)**2).sum(dim=-1)
            pyro.factor("coherence", -penalty*100)

if __name__ == "__main__":

    device = "cuda:0"
    B = 10
    H = 64

    num_steps = 1000
    lr = 1e-2

    # Example: dummy inputs
    x_noisy = torch.randn(B, H, device=device)

    model = FactorModel(X_noisy=x_noisy)
    guide = Guide(X_noisy=x_noisy)

    optimizer = SVI(model, guide, ClippedAdam({"lr": lr}), Trace_ELBO())
    start = guide.loc.detach().clone()

    for _ in range(num_steps):
        loss = optimizer.step()
        if _ % 500 == 0:
            print(f"Step {_} loss :", loss)
            # print("guide grad", guide.loc.grad)
    print(guide.loc.detach().clone())
