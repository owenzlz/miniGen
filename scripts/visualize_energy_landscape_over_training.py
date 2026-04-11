"""
Visualize the learned energy landscape over training iterations.

Trains an Energy-Based Model from scratch and periodically saves heatmaps
of the energy landscape on a 2D grid, then stitches frames into an MP4 video.

Output structure:
    assets/energy_landscape/{dataset_name}/{paradigm_type}/
        images/                     (per-frame PNG files)
        {paradigm_type}_{dataset_name}_energy_landscape.mp4

Usage:
    python scripts/visualize_energy_landscape_over_training.py
    python scripts/visualize_energy_landscape_over_training.py --total_steps 10000 --grid_size 100
    python scripts/visualize_energy_landscape_over_training.py --fps 5 --num_frames 80
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
from scripts.utils import PARADIGM_TO_CONFIG, compute_save_steps
from utils import (
    create_model,
    create_generative_process,
    create_optimizer,
    get_target_samples,
    images_to_video,
)


def compute_energy(model, grid_points, device):
    """Compute scalar energy on grid points.

    Args:
        model: Energy network E(x) -> (B,).
        grid_points: Tensor of shape (N, 2) in normalized [-1, 1] space.
        device: Torch device.

    Returns:
        Energy values as numpy array of shape (N,).
    """
    grid = grid_points.to(device)
    energy = model(grid)
    return energy.detach().cpu().numpy()


def visualize_energy_landscape(
    energy_values,
    grid_x,
    grid_y,
    target_samples,
    save_path,
    title="",
    data_min=None,
    data_range=None,
    vmin=None,
    vmax=None,
):
    """Save a heatmap of the energy landscape with target data as overlay.

    Args:
        energy_values: Energy array of shape (grid_size*grid_size,).
        grid_x: Meshgrid X coordinates in denorm space, shape (grid_size, grid_size).
        grid_y: Meshgrid Y coordinates in denorm space, shape (grid_size, grid_size).
        target_samples: Target data points (N, 2) in normalized [-1, 1] space.
        save_path: Path to save the figure.
        title: Plot title.
        data_min: Min values for denormalization.
        data_range: Range values for denormalization.
        vmin: Min log-energy for colorbar (for consistent scale across frames).
        vmax: Max log-energy for colorbar.
    """
    grid_size = grid_x.shape[0]
    energy_grid = energy_values.reshape(grid_size, grid_size)

    # Denormalize target samples for overlay
    target_denorm = target_samples
    if data_min is not None and data_range is not None:
        target_denorm = (target_samples + 1) / 2 * data_range + data_min

    # Clip energy at a moderate percentile to focus colormap on the data region.
    # Raw energy spans [~0, ~5+] where the data valley contrast (~0.06) is
    # invisible against the far-from-data tail. Clipping removes the tail so
    # the full colormap resolves the near-data structure (spiral arms vs gaps).
    plot_values = -energy_grid

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.pcolormesh(
        grid_x, grid_y, plot_values,
        cmap='inferno', shading='auto',
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label='-Energy  (high = likely)')

    # Overlay: black scatter of target data
    ax.scatter(
        target_denorm[:, 0], target_denorm[:, 1],
        s=2, alpha=0.3, color='black', edgecolors='none', zorder=2,
    )

    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_aspect('equal')
    ax.set_title(title)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned energy landscape over training iterations."
    )
    parser.add_argument(
        "--total_steps", type=int, default=20000,
        help="Number of training iterations (default: 20000).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=60,
        help="Total number of frames to save (default: 60).",
    )
    parser.add_argument(
        "--fps", type=int, default=3,
        help="Frames per second for the output video (default: 3).",
    )
    parser.add_argument(
        "--grid_size", type=int, default=80,
        help="Number of grid points per axis for heatmap (default: 80).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: use PARADIGM_TO_CONFIG for energy_based_model).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets",
        help="Root output directory (default: assets).",
    )
    parser.add_argument(
        "--smooth_clip_region", action="store_true",
        help="Smoothly fade the clipped far-from-data region to black instead of a hard cutoff.",
    )
    args = parser.parse_args()

    paradigm_type = "energy_based_model"

    # Load config
    if args.config is not None:
        config_path = PROJECT_ROOT / args.config
    else:
        config_path = PROJECT_ROOT / PARADIGM_TO_CONFIG[paradigm_type]
    cfg = OmegaConf.load(config_path)
    cfg.training.total_steps = args.total_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg.data.distribution

    # Output paths (use exp_name to distinguish configs)
    exp_name = cfg.training.get("exp_name", paradigm_type)
    output_dir = PROJECT_ROOT / args.output_dir / "energy_landscape" / dataset_name / exp_name
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{exp_name}_{dataset_name}_energy_landscape.mp4"

    # Compute which steps to save (quadratic schedule: dense early, sparse later)
    save_steps = compute_save_steps(args.total_steps, args.num_frames)

    print(f"Paradigm:    {paradigm_type}")
    print(f"Dataset:     {dataset_name}")
    print(f"Steps:       {args.total_steps}")
    print(f"Frames:      {len(save_steps)} (saving more frequently early)")
    print(f"Grid:        {args.grid_size}x{args.grid_size}")
    print(f"Output:      {output_dir}")

    # Create model, process, optimizer, dataloader
    model = create_model(cfg, device=device, inference=False)
    process = create_generative_process(cfg, device)
    optimizer = create_optimizer(model, cfg)
    dataloader = create_dataloader(cfg)

    # Pre-fetch target samples for overlay scatter
    target_samples, _, data_min, data_range = get_target_samples(
        dataloader, 1000, key="tenPoints"
    )
    if target_samples is not None:
        target_samples = target_samples.numpy()

    # Build evaluation grid in denorm space [-4, 4], then normalize to [-1, 1]
    coords = np.linspace(-4, 4, args.grid_size)
    gx, gy = np.meshgrid(coords, coords)
    grid_denorm = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (G*G, 2)

    # Normalize: x_norm = 2 * (x - data_min) / data_range - 1
    if data_min is not None and data_range is not None:
        grid_norm = 2 * (grid_denorm - data_min) / data_range - 1
    else:
        grid_norm = grid_denorm
    grid_tensor = torch.tensor(grid_norm, dtype=torch.float32)

    grad_clip = cfg.training.get("grad_clip", 1.0)

    # Training + visualization loop
    model.train()
    data_iter = iter(dataloader)
    frame_idx = 0

    for step in tqdm(range(args.total_steps), desc=f"Training {paradigm_type}"):
        # --- Get batch ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x0 = batch["tenPoints"].to(device)
        y = batch.get("intLabel")
        if y is not None:
            y = y.to(device)

        # --- Training step ---
        t = process.sample_timesteps(x0.shape[0], device)
        loss = process.training_loss(model, x0, t, y=y)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # --- Save frame at scheduled steps ---
        if step not in save_steps:
            continue

        model.eval()
        with torch.no_grad():
            energy_values = compute_energy(model, grid_tensor, device)

        # Clip-based range: cap energy at a moderate percentile so the colormap
        # focuses on the data region (valley + transition) rather than the
        # far-out-of-distribution tail that dominates the raw range.
        # p50 keeps the full near-data structure while cutting the long tail.
        clip_hi = np.percentile(energy_values, 50)

        if args.smooth_clip_region:
            # Smooth transition: compress the far-from-data region (energy > p50)
            # into a narrow band so the colormap fades gradually to black instead
            # of a hard cutoff.  The data region is preserved exactly.
            core_range = clip_hi - energy_values.min()
            transition = core_range * 0.3  # transition zone = 30% of core range

            plot_energy = energy_values.copy()
            mask = plot_energy > clip_hi
            if mask.any():
                far_max = plot_energy[mask].max()
                span = max(far_max - clip_hi, 1e-8)
                # Linear compression: [clip_hi, far_max] -> [clip_hi, clip_hi + transition]
                t = (plot_energy[mask] - clip_hi) / span
                plot_energy[mask] = clip_hi + t * transition

            neg_energy = -plot_energy
            vmin, vmax = neg_energy.min(), neg_energy.max()
        else:
            energy_clipped = np.clip(energy_values, energy_values.min(), clip_hi)
            neg_energy = -energy_clipped
            vmin, vmax = neg_energy.min(), neg_energy.max()
            plot_energy = energy_values

        save_path = img_dir / f"frame_{frame_idx:07d}.png"
        visualize_energy_landscape(
            energy_values=plot_energy,
            grid_x=gx,
            grid_y=gy,
            target_samples=target_samples,
            save_path=save_path,
            title=f"Energy landscape at iter {step}",
            data_min=data_min,
            data_range=data_range,
            vmin=vmin,
            vmax=vmax,
        )
        frame_idx += 1
        model.train()

    # Stitch frames into video
    print(f"Stitching {frame_idx} frames into video at {args.fps} fps ...")
    images_to_video(img_dir, video_path, fps=args.fps, filename_pattern="frame_%07d.png")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
