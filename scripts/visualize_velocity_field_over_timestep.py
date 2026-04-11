"""
Visualize the learned velocity field over inference timesteps.

Trains a velocity-based model from scratch, then sweeps over timesteps from
t=1 (noise) to t=0 (data) to show how the velocity field varies with time.

Supported paradigms:
    - flow_matching: velocity field v(x, t)
    - continuous_normalizing_flow: velocity field v(x, t)
    - improved_meanflow: average velocity u(x, t, h) with configurable h

Output structure:
    assets/velocity_field_over_timestep/{dataset_name}/{paradigm_type}/
        images/                     (per-frame PNG files)
        {paradigm_type}_{dataset_name}_velocity_field_over_timestep.mp4

Usage:
    python scripts/visualize_velocity_field_over_timestep.py --type flow_matching
    python scripts/visualize_velocity_field_over_timestep.py --type continuous_normalizing_flow --num_frames 50
    python scripts/visualize_velocity_field_over_timestep.py --type improved_meanflow --eval_h 0.01
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from data import create_dataloader
from scripts.utils import PARADIGM_TO_CONFIG
from utils import (
    compute_axis_limits,
    create_model,
    create_generative_process,
    create_optimizer,
    get_target_samples,
    images_to_video,
)

VELOCITY_PARADIGMS = ["flow_matching", "continuous_normalizing_flow", "improved_meanflow"]


def get_inference_timesteps(num_frames):
    """Get ordered timesteps for inference visualization (t=1 noise -> t=0 data).

    Returns:
        List of (timestep_value, label_string) tuples.
    """
    times = np.linspace(1.0, 0.0, num_frames)
    return [(float(t), f"t={t:.3f}") for t in times]


def compute_velocity_at_timestep(model, grid_points, paradigm_type, timestep, device, eval_h=None):
    """Compute velocity vectors at a specific timestep.

    Args:
        model: Trained velocity model.
        grid_points: Tensor of shape (N, 2) in normalized [-1, 1] space.
        paradigm_type: One of VELOCITY_PARADIGMS.
        timestep: Float in [0, 1].
        device: Torch device.
        eval_h: For improved_meanflow, the h (step size) to use.

    Returns:
        Velocity vectors as numpy array of shape (N, 2), negated to show
        the sampling direction (toward data).
    """
    grid = grid_points.to(device)
    batch_size = grid.shape[0]
    t = torch.full((batch_size,), timestep, device=device, dtype=torch.float32)

    if paradigm_type in ("flow_matching", "continuous_normalizing_flow"):
        velocity = model(grid, t)

    elif paradigm_type == "improved_meanflow":
        h = torch.full((batch_size,), eval_h, device=device, dtype=torch.float32)
        u, _ = model(grid, t, h)
        velocity = u

    else:
        raise ValueError(f"Unsupported paradigm: {paradigm_type}")

    # Negate to show sampling direction (noise -> data, toward data)
    return -velocity.detach().cpu().numpy()


def visualize_velocity_field(
    velocity_vectors,
    grid_x,
    grid_y,
    target_samples,
    save_path,
    title="",
    data_min=None,
    data_range=None,
):
    """Save a quiver plot of the velocity field with target data as background.

    Args:
        velocity_vectors: Velocity array of shape (grid_size*grid_size, 2) in normalized space.
        grid_x: Meshgrid X coordinates in denorm space, shape (grid_size, grid_size).
        grid_y: Meshgrid Y coordinates in denorm space, shape (grid_size, grid_size).
        target_samples: Target data points (N, 2) in normalized [-1, 1] space.
        save_path: Path to save the figure.
        title: Plot title.
        data_min: Min values for denormalization.
        data_range: Range values for denormalization.
    """
    grid_size = grid_x.shape[0]

    # Denormalize velocity vectors to denorm space (scale only, not shift)
    u = velocity_vectors[:, 0].reshape(grid_size, grid_size)
    v = velocity_vectors[:, 1].reshape(grid_size, grid_size)
    if data_range is not None:
        u = u * data_range[0] / 2
        v = v * data_range[1] / 2

    # Denormalize target samples for background
    target_denorm = target_samples
    if data_min is not None and data_range is not None:
        target_denorm = (target_samples + 1) / 2 * data_range + data_min

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Background: faint gray scatter of target data
    ax.scatter(
        target_denorm[:, 0], target_denorm[:, 1],
        s=2, alpha=0.15, color='gray', zorder=1,
    )

    # Foreground: blue quiver arrows for velocity vectors
    ax.quiver(
        grid_x, grid_y, u, v,
        color='blue', alpha=0.8, scale=None, zorder=2,
    )

    xlim, ylim = compute_axis_limits(data_min, data_range)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned velocity field over inference timesteps."
    )
    parser.add_argument(
        "--type", type=str, required=True,
        choices=VELOCITY_PARADIGMS,
        help="Velocity-based paradigm type.",
    )
    parser.add_argument(
        "--total_steps", type=int, default=5000,
        help="Number of training iterations (default: 5000).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=60,
        help="Number of timestep frames to visualize (default: 60).",
    )
    parser.add_argument(
        "--fps", type=int, default=3,
        help="Frames per second for the output video (default: 3).",
    )
    parser.add_argument(
        "--grid_size", type=int, default=20,
        help="Number of grid points per axis for quiver plot (default: 20).",
    )
    parser.add_argument(
        "--eval_h", type=float, default=None,
        help="For improved_meanflow: h (step size) for velocity evaluation. "
             "Default: 0.01 (approximate instantaneous velocity).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets",
        help="Root output directory (default: assets).",
    )
    args = parser.parse_args()

    # Set default eval_h for iMF
    if args.eval_h is None:
        args.eval_h = 0.01

    # Load default config for this paradigm
    config_path = PROJECT_ROOT / PARADIGM_TO_CONFIG[args.type]
    cfg = OmegaConf.load(config_path)
    cfg.training.total_steps = args.total_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg.data.distribution
    paradigm_type = cfg.type

    # Output paths
    output_dir = PROJECT_ROOT / args.output_dir / "velocity_field_over_timestep" / dataset_name / paradigm_type
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{paradigm_type}_{dataset_name}_velocity_field_over_timestep.mp4"

    # Label for what we're visualizing
    if paradigm_type == "improved_meanflow":
        field_label = f"avg velocity u (h={args.eval_h})"
    else:
        field_label = "velocity field"

    print(f"Paradigm:    {paradigm_type}")
    print(f"Dataset:     {dataset_name}")
    print(f"Training:    {args.total_steps} steps")
    print(f"Frames:      {args.num_frames}")
    print(f"Grid:        {args.grid_size}x{args.grid_size}")
    print(f"Field:       {field_label}")
    print(f"Output:      {output_dir}")

    # Create model, process, optimizer, dataloader
    model = create_model(cfg, device=device, inference=False)
    process = create_generative_process(cfg, device)
    optimizer = create_optimizer(model, cfg)
    dataloader = create_dataloader(cfg)

    # Pre-fetch target samples for background scatter
    target_samples, _, data_min, data_range = get_target_samples(
        dataloader, 1000, key="tenPoints"
    )
    if target_samples is not None:
        target_samples = target_samples.numpy()

    # Build evaluation grid in denorm space, then normalize to [-1, 1]
    xlim, ylim = compute_axis_limits(data_min, data_range)
    coords_x = np.linspace(xlim[0], xlim[1], args.grid_size)
    coords_y = np.linspace(ylim[0], ylim[1], args.grid_size)
    gx, gy = np.meshgrid(coords_x, coords_y)
    grid_denorm = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (G*G, 2)

    if data_min is not None and data_range is not None:
        grid_norm = 2 * (grid_denorm - data_min) / data_range - 1
    else:
        grid_norm = grid_denorm
    grid_tensor = torch.tensor(grid_norm, dtype=torch.float32)

    grad_clip = cfg.training.get("grad_clip", 1.0)

    # ---- Phase 1: Train the model ----
    print(f"\nPhase 1: Training {paradigm_type} for {args.total_steps} steps ...")
    model.train()
    data_iter = iter(dataloader)

    for step in tqdm(range(args.total_steps), desc=f"Training {paradigm_type}"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x0 = batch["tenPoints"].to(device)
        y = batch.get("intLabel")
        if y is not None:
            y = y.to(device)

        t = process.sample_timesteps(x0.shape[0], device)
        loss = process.training_loss(model, x0, t, y=y)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # ---- Phase 2: Visualize velocity field over inference timesteps ----
    inference_timesteps = get_inference_timesteps(args.num_frames)

    print(f"\nPhase 2: Visualizing {field_label} over {len(inference_timesteps)} timesteps ...")

    model.eval()
    for frame_idx, (ts_val, ts_label) in enumerate(
        tqdm(inference_timesteps, desc=f"Visualizing {field_label}")
    ):
        with torch.no_grad():
            velocity_vectors = compute_velocity_at_timestep(
                model, grid_tensor, paradigm_type, ts_val, device,
                eval_h=args.eval_h,
            )

        save_path = img_dir / f"frame_{frame_idx:07d}.png"
        visualize_velocity_field(
            velocity_vectors=velocity_vectors,
            grid_x=gx,
            grid_y=gy,
            target_samples=target_samples,
            save_path=save_path,
            title=f"{paradigm_type} {field_label} at {ts_label}",
            data_min=data_min,
            data_range=data_range,
        )

    # Stitch frames into video
    num_frames_actual = len(inference_timesteps)
    print(f"\nStitching {num_frames_actual} frames into video at {args.fps} fps ...")
    images_to_video(img_dir, video_path, fps=args.fps, filename_pattern="frame_%07d.png")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
