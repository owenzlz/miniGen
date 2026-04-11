"""
Training script for generative models.

Usage:
    # DDPM / DDIM
    python train.py --config configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml
    python train.py --config configs/diffusion/DDIMSampler_MLP_SwissRoll.yaml

    # Flow Matching
    python train.py --config configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml

    # VAE / GAN / Score Matching (NCSN) / Normalizing Flow / CNF
    python train.py --config configs/variational_autoencoder/VAE_MLP_SwissRoll.yaml
    python train.py --config configs/generative_adversarial_network/GAN_MLP_SwissRoll.yaml
    python train.py --config configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_SwissRoll.yaml
    python train.py --config configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_SwissRoll.yaml
    python train.py --config configs/normalizing_flow/NormalizingFlow_RealNVP_SwissRoll.yaml
    python train.py --config configs/continuous_normalizing_flow/CNF_MLP_SwissRoll.yaml
    python train.py --config configs/consistency_training/ConsistencyTraining_MLP_SwissRoll.yaml
    python train.py --config configs/energy_based_model/EBM_MLP_SwissRoll.yaml
"""
import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

from data import create_dataloader
from utils import (
    create_model,
    create_generative_process,
    create_optimizer,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
    sample,
    get_target_samples,
    visualize_synthetic2d,
    update_visualization_html,
)


@torch.no_grad()
def generate_and_visualize(
    model,
    process,
    cfg,
    device,
    step,
    log_dir,
    dataloader=None,
    sampler=None,
    num_steps=None,
):
    """Generate samples and save visualizations.

    Args:
        model: The model.
        process: Generative process (DDPM or FlowMatching).
        cfg: Config object.
        device: Torch device.
        step: Current training step.
        log_dir: Directory to save visualizations.
        dataloader: Optional dataloader for target distribution comparison.
        sampler: Optional sampler (DDPMSampler or DDIMSampler).
        num_steps: Optional number of sampling steps.

    Returns:
        Path to saved visualization.
    """
    model.eval()

    img_dir = log_dir / "saved_images"
    save_path = img_dir / f"samples_{step:07d}.png"

    num_samples = 1000
    shape = (num_samples, cfg.model.get("input_dim", 2))
    num_classes = cfg.model.get("num_classes", 0)

    # Generate class labels if class-conditional
    y = None
    if num_classes > 0:
        y = torch.randint(0, num_classes, (num_samples,), device=device)

    # Generate samples
    samples = sample(
        model=model,
        process=process,
        shape=shape,
        device=device,
        sampler=sampler,
        num_steps=num_steps,
        class_labels=y,
        use_tqdm=False,
    )

    # Get target samples for comparison
    target_samples, target_labels, data_min, data_range = get_target_samples(
        dataloader, num_samples, key="tenPoints"
    )
    if target_samples is not None:
        target_samples = target_samples.numpy()
    if target_labels is not None:
        target_labels = target_labels.numpy()

    # Visualize
    save_path = visualize_synthetic2d(
        samples=samples,
        save_path=save_path,
        step=step,
        labels=y,
        target_samples=target_samples,
        target_labels=target_labels,
        data_min=data_min,
        data_range=data_range,
        num_classes=num_classes,
    )

    model.train()
    return save_path


def train(cfg, skip_save_ckpt=False):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{cfg.training.exp_name}"
    log_dir = Path(cfg.training.log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")
    print(f"Logging to: {log_dir}")

    # Create model
    model = create_model(cfg, device=device, inference=False)

    # Create generative process (DDPM or FlowMatching)
    process = create_generative_process(cfg, device)

    # Create EMA (optional)
    ema_decay = cfg.training.get("ema_decay", 0.0)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema:
        print(f"Using EMA with decay={ema_decay}")

    # Create optimizer
    optimizer = create_optimizer(model, cfg)

    # Resume from checkpoint
    start_step = 0
    if cfg.training.get("resume"):
        start_step = load_checkpoint(cfg.training.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # Create dataloader
    dataloader = create_dataloader(cfg)

    # Give the process access to the dataloader (used by AE for reconstruction sampling)
    if hasattr(process, "set_dataloader"):
        process.set_dataloader(dataloader)

    # Training settings
    total_steps = cfg.training.total_steps
    log_freq = cfg.training.get("log_freq", 100)
    save_freq = cfg.training.get("save_freq", 10000)
    vis_freq = cfg.training.get("vis_freq", 0)  # 0 = disabled
    grad_clip = cfg.training.get("grad_clip", 1.0)

    # Initialize training logger
    logger = TrainingLogger(
        log_dir=log_dir,
        total_steps=total_steps,
        exp_name=cfg.training.exp_name,
    )

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    pbar = tqdm(range(start_step, total_steps), desc="Training")

    for step in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x0 = batch["tenPoints"].to(device)
        y = batch.get("intLabel")
        if y is not None:
            y = y.to(device)

        # Training step
        if hasattr(process, 'train_step'):
            # GAN-style: process manages both G and D updates
            loss_val = process.train_step(model, x0, optimizer, y=y, grad_clip=grad_clip)
        else:
            # Standard: process computes loss, train.py handles backward/step
            t = process.sample_timesteps(x0.shape[0], device)
            loss = process.training_loss(model, x0, t, y=y)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_val = loss.item()

        # EMA update
        if ema is not None:
            ema.update(model)

        # Logging
        if step % log_freq == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            lr = optimizer.param_groups[0]["lr"]
            logger.log(step, loss_val, lr)

        # Save checkpoint
        if not skip_save_ckpt and step > 0 and step % save_freq == 0:
            ckpt_dir = log_dir / "saved_checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            save_checkpoint(model, optimizer, step, cfg, ckpt_dir / f"checkpoint_{step:07d}.pt")

        # Visualize samples
        if vis_freq > 0 and step > 0 and step % vis_freq == 0:
            vis_model = ema.shadow if ema is not None else model
            vis_path = generate_and_visualize(vis_model, process, cfg, device, step, log_dir, dataloader=dataloader)
            update_visualization_html(log_dir)
            tqdm.write(f"Saved samples to {vis_path}")

    # Final save, visualization, and logging
    if not skip_save_ckpt:
        ckpt_dir = log_dir / "saved_checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        save_checkpoint(model, optimizer, total_steps, cfg, ckpt_dir / "checkpoint_final.pt")
        if ema is not None:
            torch.save(ema.state_dict(), ckpt_dir / "ema_final.pt")
    if vis_freq > 0:
        vis_model = ema.shadow if ema is not None else model
        generate_and_visualize(vis_model, process, cfg, device, total_steps, log_dir, dataloader=dataloader)
        update_visualization_html(log_dir)
    logger.finalize()


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--skip_save_ckpt", action="store_true", help="Skip saving checkpoints")
    parser.add_argument("opts", nargs="*", help="Override config options")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override with command line options
    if args.opts:
        cli_cfg = OmegaConf.from_dotlist(args.opts)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    train(cfg, skip_save_ckpt=args.skip_save_ckpt)


if __name__ == "__main__":
    main()
