"""
Child Class from Gaussian Diffusion Model
Adding Factor Graph inference steps between denoising 
"""
import torch

from mpd.models.diffusion_models import GaussianDiffusionModel
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, guide_gradient_steps, \
    ddpm_sample_fn

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
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None

        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, hard_conds, context, t, **sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False, sample_fn=..., n_diffusion_steps_without_noise=0, **sample_kwargs):
        """
            Add factor graph inference during sampling step
            # TODO: add inference steps and inference functions here
        """
        return super().p_sample_loop(shape, hard_conds, context, return_chain, sample_fn, n_diffusion_steps_without_noise, **sample_kwargs)()