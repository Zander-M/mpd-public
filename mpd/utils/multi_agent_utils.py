"""
    Multi-agent utils for experiments.
    Functions borrowed from MMD
"""

import numpy as np
import torch

def get_start_goal_pos_circle(num_agents: int, radius=0.8):
    """
        Generate circular start-goal pos
    """
    # These are all in the local tile frame.
    start_l = [torch.tensor([radius * np.cos(2 * torch.pi * i / num_agents),
                             radius * np.sin(2 * torch.pi * i / num_agents)],
                            dtype=torch.float32, device='cuda') for i in range(num_agents)]
    goal_l = [torch.tensor([radius * np.cos(2 * torch.pi * i / num_agents + torch.pi),
                            radius * np.sin(2 * torch.pi * i / num_agents + torch.pi)],
                           dtype=torch.float32, device='cuda') for i in range(num_agents)]
    return start_l, goal_l

