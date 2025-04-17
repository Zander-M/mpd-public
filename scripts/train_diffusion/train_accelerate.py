import os
import torch
from accelerate import Accelerator

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, TemporalUnet
from mpd.trainer import get_dataset, get_model, get_loss, get_summary
from mpd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
accelerator = Accelerator()


@single_experiment_yaml
def experiment(
    dataset_subdir: str = 'EnvEmpty2D-RobotPointMass',
    include_velocity: bool = True,
    diffusion_model_class: str = 'GaussianDiffusionFactorModel',
    variance_schedule: str = 'exponential',
    n_diffusion_steps: int = 25,
    predict_epsilon: bool = True,
    unet_input_dim: int = 32,
    unet_dim_mults_option: int = 1,
    loss_class: str = 'GaussianDiffusionLoss',
    batch_size: int = 512,
    lr: float = 4e-4,
    num_train_steps: int = 50000,
    use_ema: bool = True,
    use_amp: bool = False,
    steps_til_summary: int = 1000,
    summary_class: str = 'SummaryTrajectoryGeneration',
    steps_til_ckpt: int = 5000,
    # device: str = 'cuda',
    debug: bool = True,
    seed: int = 0,
    results_dir: str = 'logs',
    wandb_mode: str = 'disabled',
    wandb_entity: str = 'scoreplan',
    wandb_project: str = 'train_diffusion',
    **kwargs
):
    fix_random_seed(seed)
    print(f"[Rank {accelerator.process_index}] Using device: {accelerator.device}")
    # Set up Accelerate
    device = accelerator.device
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=include_velocity,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
    )
    dataset = train_subset.dataset

    # Model
    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
    )

    model = get_model(
        model_class=diffusion_model_class,
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    loss_fn = val_loss_fn = get_loss(loss_class=loss_class)

    # Summary
    summary_fn = get_summary(summary_class=summary_class)

    # Prepare all for accelerate
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        epochs=get_num_epochs(num_train_steps, batch_size, len(dataset)),
        model_dir=results_dir,
        summary_fn=summary_fn,
        lr=lr,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=True,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args,
        accelerator=accelerator  # pass for logging/barrier/etc.
    )


if __name__ == '__main__':
    run_experiment(experiment)
