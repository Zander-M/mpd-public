import torch
from matplotlib import pyplot as plt


def apply_hard_conditioning(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def ddpm_sample_fn(
        model, x, hard_conds, context, t,
        guide=None,
        n_guide_steps=1,
        scale_grad_by_std=False,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        debug=False,
        **kwargs
):
    t_single = t[0]
    if t_single < 0:
        t = torch.zeros_like(t)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, hard_conds=hard_conds, context=context, t=t)
    x = model_mean

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    if guide is not None and t_single < t_start_guide:
        x = guide_gradient_steps(
            x,
            hard_conds=hard_conds,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            model_var=model_var,
            debug=False,
        )

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    values = None
    return x + model_std * noise * noise_std, values


def guide_gradient_steps(
    x,
    hard_conds=None,
    guide=None,
    n_guide_steps=1, scale_grad_by_std=False,
    model_var=None,
    debug=False,
    **kwargs
):
    for _ in range(n_guide_steps):
        grad_scaled = guide(x)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        x = x + grad_scaled
        x = apply_hard_conditioning(x, hard_conds)

    return x

## Factor Guided DDPM sampling function, with SVI factor update on predicted data
@torch.no_grad()
def ddpm_sample_fn_with_factor_guidance(
        model, x, hard_conds, context, t,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        **kwargs
):

    assert kwargs["guidance"] is not None
    guidance = kwargs["guidance"]
    guidance_param = kwargs["guidance_param"]
    t_single = t[0]
    if t_single < 0:
        t = torch.zeros_like(t)

    # Predict the clean reconstruction x0_hat
    if context is not None:
        context = model.context_model(context)
    
    noise_pred = model.model(x, t, context)
    x_recon = model.predict_start_from_noise(x, t=t, noise=noise_pred)

    if model.clip_denoised:
        x_recon.clamp_(-1., 1.)

    # Apply Model-Based inference
    B = kwargs["guidance_param"]["B"]
    N = kwargs["guidance_param"]["N"]
    H = kwargs["guidance_param"]["H"]
    D = kwargs["guidance_param"]["D"]
    x_recon = x_recon.view(B, N, H, D)
    if t_single < t_start_guide:
        alpha_bar_t = model.alphas_cumprod[t_single].clone().detach()
        with torch.enable_grad():
            guidance = guidance(**guidance_param)
            x_recon_refined = guidance.forward(x_recon, alpha_bar_t)
        if kwargs["debug"]:
            print(f"Denoising step {t_single}", guidance.compute_refinement(x_recon, x_recon_refined))
    else:
        x_recon_refined = x_recon

    x_recon_refined = x_recon_refined.view(B*N, H, D).clone().detach()

        
    # Compute posterior q(x_{t-1} | x_t, x0_hat)
    model_mean, posterior_variance, posterior_log_vaiance = model.q_posterior(
        x_start = x_recon_refined,
        x_t = x,
        t = t
    )
    model_std = torch.exp(0.5 * posterior_log_vaiance)

    # Adding noise back
    noise = torch.randn_like(x)
    noise[t==0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is roughly equivalent to using a temperature in the prior distribution
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    values = None
    return model_mean + model_std * noise * noise_std, values

