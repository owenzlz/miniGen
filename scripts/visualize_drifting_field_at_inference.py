"""
Visualize the one-step inference of a trained Drifting Model.

Trains a Drifting Model (or loads a checkpoint), then shows how random
noise z is mapped to data-space points in a single forward pass.

The animation smoothly interpolates from noise to output:
  - Blue dots: random noise z (projected to 2D via PCA for display)
  - Green lines: interpolated paths from noise to output
  - Red dots: generator output model(z) on the data manifold

Output structure:
    assets/drifting_field/{dataset_name}/drifting_model/
        images/                     (per-frame PNG files)
        drifting_model_{dataset_name}_drifting_field.mp4

Usage:
    # Train from scratch and save checkpoint:
    python scripts/visualize_drifting_field_over_training.py

    # Load checkpoint (skip training) and tweak visualization:
    python scripts/visualize_drifting_field_over_training.py --ckpt ckpt/drifting_model.pt
"""
import argparse
import shutil
import subprocess
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
    create_model,
    create_generative_process,
    create_optimizer,
    get_target_samples,
    images_to_video,
)


def visualize_drift_frame(
    start_pts,
    end_pts,
    frac,
    target_samples,
    save_path,
    title="",
    data_min=None,
    data_range=None,
):
    """Save one frame of the one-step mapping animation.

    Args:
        start_pts: Starting 2D positions (N, 2) in normalized space.
        end_pts: Ending 2D positions (N, 2) in normalized space (model output).
        frac: Interpolation fraction in [0, 1]. 0=start, 1=end.
        target_samples: Target data points (N, 2) in normalized [-1, 1] space.
        save_path: Path to save the figure.
        title: Plot title.
        data_min: Min values for denormalization.
        data_range: Range values for denormalization.
    """
    def denorm(pts):
        if data_min is not None and data_range is not None:
            return (pts + 1) / 2 * data_range + data_min
        return pts

    target_denorm = denorm(target_samples)
    start_denorm = denorm(start_pts)
    end_denorm = denorm(end_pts)
    current_denorm = (1 - frac) * start_denorm + frac * end_denorm

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Background: faint gray scatter of target data
    ax.scatter(
        target_denorm[:, 0], target_denorm[:, 1],
        s=2, alpha=0.15, color='gray', zorder=1,
    )

    # Green lines from start to current position
    num_points = start_denorm.shape[0]
    for i in range(num_points):
        ax.plot(
            [start_denorm[i, 0], current_denorm[i, 0]],
            [start_denorm[i, 1], current_denorm[i, 1]],
            color='green', alpha=0.4, linewidth=0.8, zorder=2,
        )

    # Light green dots: starting noise positions
    ax.scatter(
        start_denorm[:, 0], start_denorm[:, 1],
        s=15, color='blue', alpha=0.7, zorder=3, label='Noise z',
    )

    # Dark green dots: current interpolated positions
    ax.scatter(
        current_denorm[:, 0], current_denorm[:, 1],
        s=15, color='red', alpha=0.7, zorder=4, label='G(z)',
    )

    # Data-adaptive axis limits: center on target data, scale ~2x its extent
    center = np.mean(target_denorm, axis=0)
    target_half_span = np.max(np.abs(target_denorm - center))
    half_span = target_half_span * 2.0
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize one-step drifting model inference."
    )
    parser.add_argument(
        "--total_steps", type=int, default=5000,
        help="Number of training iterations (default: 5000).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=60,
        help="Number of animation frames (default: 60).",
    )
    parser.add_argument(
        "--num_points", type=int, default=300,
        help="Number of random noise samples (default: 300).",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for the output video (default: 10).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: use PARADIGM_TO_CONFIG for drifting_model).",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to checkpoint to load (skip training).",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="ckpt",
        help="Directory to save checkpoint after training (default: ckpt).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets",
        help="Root output directory (default: assets).",
    )
    parser.add_argument(
        "--distribution", type=str, default=None,
        help="Override data distribution in config.",
    )
    args = parser.parse_args()

    paradigm_type = "drifting_model"

    # Load config
    if args.config is not None:
        config_path = PROJECT_ROOT / args.config
    else:
        config_path = PROJECT_ROOT / PARADIGM_TO_CONFIG[paradigm_type]
    cfg = OmegaConf.load(config_path)
    cfg.training.total_steps = args.total_steps
    if args.distribution is not None:
        cfg.data.distribution = args.distribution
        # gaussian_mixture needs num_classes > 0 for mode assignment
        if args.distribution == "gaussian_mixture" and cfg.model.get("num_classes", 0) == 0:
            cfg.model.num_classes = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg.data.distribution

    # Output paths
    output_dir = PROJECT_ROOT / args.output_dir / "drifting_field" / dataset_name / paradigm_type
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{paradigm_type}_{dataset_name}_drifting_field.mp4"

    print(f"Paradigm:    {paradigm_type}")
    print(f"Dataset:     {dataset_name}")
    print(f"Num points:  {args.num_points}")
    print(f"Num frames:  {args.num_frames}")
    print(f"Output:      {output_dir}")

    # Create model, process, dataloader
    model = create_model(cfg, device=device, inference=False)
    process = create_generative_process(cfg, device)
    dataloader = create_dataloader(cfg)

    # Pre-fetch target samples for background scatter
    target_samples, _, data_min, data_range = get_target_samples(
        dataloader, 2000, key="tenPoints"
    )
    target_np = target_samples.numpy()

    # ---- Phase 1: Train or load checkpoint ----
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"drifting_model_{dataset_name}.pt"

    if args.ckpt is not None:
        load_path = Path(args.ckpt)
        if not load_path.is_absolute():
            load_path = PROJECT_ROOT / load_path
        print(f"\nLoading checkpoint from {load_path} ...")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        optimizer = create_optimizer(model, cfg)
        grad_clip = cfg.training.get("grad_clip", 1.0)

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

        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    # ---- Phase 2: One-step inference ----
    model.eval()
    latent_dim = cfg.model.get("latent_dim", 32)

    # Sample random noise and compute generator output (one forward pass)
    z = torch.randn(args.num_points, latent_dim, device=device)
    with torch.no_grad():
        gen_output = model(z)  # (N, 2) — on the data manifold

    # For display: project z to 2D via its first 2 principal components
    # so blue dots appear as scattered random noise in the same 2D plot
    z_np = z.cpu().numpy()  # (N, latent_dim)
    z_mean = z_np.mean(axis=0)
    z_centered = z_np - z_mean
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_2d = z_centered @ Vt[:2].T  # (N, 2)

    # Scale z_2d to fill the plot area so noise looks clearly scattered
    gen_np = gen_output.cpu().numpy()
    z_scale = 3.0 * np.std(gen_np) / max(np.std(z_2d), 1e-8)
    z_2d_scaled = z_2d * z_scale

    start_pts = z_2d_scaled  # blue dots: noise (2D projection)
    end_pts = gen_np          # red dots: model output

    print(f"\nPhase 2: Rendering {args.num_frames} frames (one-step: noise → G(z)) ...")

    # ---- Phase 3: Render interpolation frames ----
    for frame_idx in tqdm(range(args.num_frames), desc="Rendering frames"):
        frac = frame_idx / max(args.num_frames - 1, 1)
        save_path = img_dir / f"frame_{frame_idx:07d}.png"
        visualize_drift_frame(
            start_pts=start_pts,
            end_pts=end_pts,
            frac=frac,
            target_samples=target_np,
            save_path=save_path,
            title=f"Drifting Model: noise → G(z)  [{frac:.0%}]",
            data_min=data_min,
            data_range=data_range,
        )

    # Duplicate last frame for 1 extra second of hold
    last_frame = img_dir / f"frame_{args.num_frames - 1:07d}.png"
    for i in range(args.fps):
        dst = img_dir / f"frame_{args.num_frames + i:07d}.png"
        shutil.copy2(last_frame, dst)

    total_frames = args.num_frames + args.fps

    # Stitch frames into video
    print(f"\nStitching {total_frames} frames into video at {args.fps} fps ...")
    images_to_video(img_dir, video_path, fps=args.fps, filename_pattern="frame_%07d.png")
    print(f"Video saved to {video_path}")

    # Generate GIF
    gif_path = video_path.with_suffix(".gif")
    print(f"Generating GIF at {gif_path} ...")
    gif_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(img_dir / "frame_%07d.png"),
        "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(gif_path),
    ]
    result = subprocess.run(gif_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"GIF generation failed: {result.stderr}")
    else:
        print(f"GIF saved to {gif_path}")


if __name__ == "__main__":
    main()
