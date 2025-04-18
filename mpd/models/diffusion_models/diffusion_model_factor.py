"""
Child Class from Gaussian Diffusion Model
Adding Factor Graph inference steps between denoising 
Provides Run Inference Once function
"""
from copy import copy

import einops
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

from mpd.models.diffusion_models import GaussianDiffusionModel
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, ddpm_sample_fn

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

def build_context(model, dataset, input_dict):
    # input_dict is already normalized
    context = None
    if model.context_model is not None:
        context = dict()
        # (normalized) features of variable environments
        if dataset.variable_environment:
            env_normalized = input_dict[f'{dataset.field_key_env}_normalized']
            context['env'] = env_normalized

        # tasks
        task_normalized = input_dict[f'{dataset.field_key_task}_normalized']
        context['tasks'] = task_normalized
    return context

class GaussianDiffusionFactorModel(GaussianDiffusionModel):

    def __init__(self, model=None, variance_schedule='exponential', n_diffusion_steps=100, clip_denoised=True, predict_epsilon=False, loss_type='l2', context_model=None, **kwargs):
        super().__init__(model, variance_schedule, n_diffusion_steps, clip_denoised, predict_epsilon, loss_type, context_model, **kwargs)

    @torch.no_grad()
    def run_inference_one_step(self, 
                               trajs_normalized,
                               t,
                               agent_idx=None,
                               context=None, hard_conds=None, 
                               n_samples=1, 
                               model_guide=None,
                               model_step=50,
                               **diffusion_kwargs):
        """
        Run one denoising step for all agents in batch mode (or one agent if agent_idx is specified).
        This version flattens (num_agents, num_samples) into a single batch dimension, and applies hard conditions
        based on trajectory index.
        """
        import copy
        trajs_normalized = trajs_normalized.clone()

        if agent_idx is not None:
            traj = trajs_normalized[agent_idx]
            hard_conds_agent = copy.copy(hard_conds[agent_idx])
            context_agent = copy.copy(context[agent_idx]) if context is not None else None

            for k, v in hard_conds_agent.items():
                hard_conds_agent[k] = einops.repeat(v, 'd -> b d', b=n_samples)
            if context_agent is not None:
                for k, v in context_agent.items():
                    context_agent[k] = einops.repeat(v, 'd -> b d', b=n_samples)

            samples = self.conditional_sample_one_step(
                traj, t, hard_conds_agent, context=context_agent, batch_size=n_samples, **diffusion_kwargs
            )

            if model_guide is not None:
                mask = torch.ones(trajs_normalized.shape[0], dtype=bool, device=trajs_normalized.device)
                mask[agent_idx] = False
                noisy_trajs_others = trajs_normalized[mask]
                alpha_bar_t = torch.tensor(self.alphas_cumprod[t], device=samples.device)
                updated_samples = model_guide.infer(
                    noisy_traj_batch=samples,
                    noisy_trajs_others=noisy_trajs_others,
                    alpha_bar_t=alpha_bar_t,
                    factor_steps=model_step,
                )
                samples[:] = updated_samples

            return samples

        else:
            B, N, H, D = trajs_normalized.shape
            trajs_flat = trajs_normalized.view(B * N, H, D)

            # Create a long list of hard_conds indexed by sample index
            # Each i-th item in hard_conds_list corresponds to sample i in trajs_flat
            cond_keys = hard_conds[0].keys()
            hard_cond_tensor = {
                k: torch.stack([h[k] for h in hard_conds], dim=0).to(trajs_normalized.device)  # (B, D)
                for k in cond_keys
            }

            # Indexing: [i * n_samples + j] corresponds to agent i, sample j
            cond_dict = {}
            for k, v in hard_cond_tensor.items():
                v_exp = einops.repeat(v, 'b d -> (b n) d', n=n_samples)  # shape (B*N, D)
                cond_dict[k] = v_exp

            # Same for context if provided
            merged_context = None
            if context is not None:
                context_keys = context[0].keys()
                context_tensor = {
                    k: torch.stack([c[k] for c in context], dim=0).to(trajs_normalized.device)
                    for k in context_keys
                }
                merged_context = {
                    k: einops.repeat(v, 'b d -> (b n) d', n=n_samples)
                    for k, v in context_tensor.items()
                }

            samples_flat = self.conditional_sample_one_step(
                trajs_flat, t, cond_dict, context=merged_context, batch_size=B * N, **diffusion_kwargs
            )

            if model_guide is not None:
                alpha_bar_t = torch.tensor(self.alphas_cumprod[t], device=samples_flat.device)
                samples_flat = model_guide.infer(
                    noisy_traj_batch=samples_flat,
                    noisy_trajs_others=None,
                    alpha_bar_t=alpha_bar_t,
                    factor_steps=model_step,
                )

            return samples_flat.view(B, N, H, D)


    @torch.no_grad()
    def conditional_sample_one_step(self, x, t, hard_conds, horizon=None, batch_size=1, **sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        return self.p_sample_one_step(x, t, hard_conds, **sample_kwargs)

    @torch.no_grad()
    def p_sample_one_step(self, x, t, hard_conds, context=None,
                      sample_fn=ddpm_sample_fn,
                      **sample_kwargs):
        """
        Perform a single denoising step at timestep t
        """
        x = x.clone()
        shape = x.shape
        device = self.betas.device
        batch_size = shape[0]

        # apply hard constraints

        x = apply_hard_conditioning(x, hard_conds)
        t = make_timesteps(batch_size, t, device)

        # One step of denoising using the sampling function
        x, values = sample_fn(self, x, hard_conds, context, t, **sample_kwargs)
        x = apply_hard_conditioning(x, hard_conds)
        return x