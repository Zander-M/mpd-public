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
                               agent_idx,
                               context=None, hard_conds=None, 
                               n_samples=1, 
                               model_guide=None,
                               model_step=50,
                               **diffusion_kwargs):
        """
            Run inference for one step for one agent, conditioned on other agent's trajectories
        """
        trajs_normalized = trajs_normalized.clone()
        traj = trajs_normalized[agent_idx]

        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        if context is not None:
            for k, v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Sample from diffusion model
        samples = self.conditional_sample_one_step(
            traj, t, hard_conds, context=context, batch_size=n_samples, **diffusion_kwargs
        )

        # Model Guidance
        if model_guide is not None:
            mask = torch.ones(trajs_normalized.shape[0], dtype=bool, device=trajs_normalized.device)
            mask[agent_idx] = False
            noisy_trajs_others = trajs_normalized[mask]  # (N-1, B, T, D)

            # prepare variance based on schedule
            alpha_bar_t = self.alphas_cumprod[t].clone().detach().to(samples.device)
            alpha_bar_t = torch.tensor(alpha_bar_t, device=samples.device)

            updated_samples = model_guide.infer(
                noisy_traj_batch=samples,
                noisy_trajs_others=noisy_trajs_others,
                alpha_bar_t=alpha_bar_t,
                factor_steps=model_step, # make this a parameter
            )
            samples[:] = updated_samples


        # return the last denoising step
        return samples

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