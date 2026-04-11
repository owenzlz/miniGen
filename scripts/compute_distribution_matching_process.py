"""
Compute the distribution matching process for a generative paradigm.

Trains the model from scratch using the default config for the specified paradigm type,
saving 60 frames with frequency inversely proportional to training step (more frames
early, fewer later). Stitches all frames into an MP4 video (~20s at 3 fps).

Output structure:
    {output_dir}/{dataset_name}/
        ground_truth.png                        (saved once if not already present)
        {paradigm_type}/
            images/                             (per-frame PNG files)
            {paradigm_type}_{dataset_name}_distribution_matching_process.mp4

Usage:
    python scripts/compute_distribution_matching_process.py --type variational_autoencoder
    python scripts/compute_distribution_matching_process.py --type ddpm --total_steps 5000 --fps 3
    python scripts/compute_distribution_matching_process.py --type drifting_model --dataset moons --hold_last_s 1
"""
import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from data import create_dataloader
from scripts.utils import PARADIGM_TO_CONFIG, compute_save_steps
from utils import (
    create_model,
    create_generative_process,
    create_optimizer,
    sample,
    get_target_samples,
    visualize_distribution,
    images_to_video,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute distribution matching process for a generative paradigm."
    )
    parser.add_argument(
        "--type", type=str, required=True,
        choices=list(PARADIGM_TO_CONFIG.keys()),
        help="Generative paradigm type.",
    )
    parser.add_argument(
        "--total_steps", type=int, default=5000,
        help="Number of training iterations (default: 5000).",
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
        "--num_samples", type=int, default=1000,
        help="Number of samples to generate per frame (default: 1000).",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override data distribution (e.g., moons, circles, spirals).",
    )
    parser.add_argument(
        "--hold_last_s", type=float, default=0,
        help="Hold the last frame for this many extra seconds (default: 0).",
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="Strip titles, grids, ticks, and axes — just the points.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets/distribution_matching_process",
        help="Base output directory (default: assets/distribution_matching_process).",
    )
    args = parser.parse_args()

    # Load default config for this paradigm
    config_path = PROJECT_ROOT / PARADIGM_TO_CONFIG[args.type]
    cfg = OmegaConf.load(config_path)
    cfg.training.total_steps = args.total_steps

    # Override dataset distribution if specified
    if args.dataset is not None:
        cfg.data.distribution = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = cfg.data.distribution  # e.g., "swiss_roll"
    paradigm_type = cfg.type              # e.g., "variational_autoencoder"

    # Output paths
    dataset_dir = PROJECT_ROOT / args.output_dir / dataset_name
    output_dir = dataset_dir / paradigm_type
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{paradigm_type}_{dataset_name}_distribution_matching_process.mp4"

    # Compute which steps to save (quadratic schedule: dense early, sparse later)
    save_steps = compute_save_steps(args.total_steps, args.num_frames)

    print(f"Paradigm:    {paradigm_type}")
    print(f"Dataset:     {dataset_name}")
    print(f"Steps:       {args.total_steps}")
    print(f"Frames:      {len(save_steps)} (saving more frequently early)")
    print(f"Output:      {output_dir}")

    # Create model, process, optimizer, dataloader
    model = create_model(cfg, device=device, inference=False)
    process = create_generative_process(cfg, device)
    optimizer = create_optimizer(model, cfg)
    dataloader = create_dataloader(cfg)

    # Pre-fetch target samples
    target_samples, target_labels, data_min, data_range = get_target_samples(
        dataloader, args.num_samples, key="tenPoints"
    )
    if target_samples is not None:
        target_samples = target_samples.numpy()
    if target_labels is not None:
        target_labels = target_labels.numpy()

    # Save ground truth distribution (once per dataset, skip if exists)
    gt_path = dataset_dir / "ground_truth.png"
    if not gt_path.exists():
        visualize_distribution(
            samples=target_samples,
            save_path=gt_path,
            title=f"Ground Truth ({dataset_name})",
            labels=target_labels,
            data_min=data_min,
            data_range=data_range,
            num_classes=cfg.model.get("num_classes", 0),
            minimal=args.minimal,
        )
        print(f"Saved ground truth: {gt_path}")

    num_classes = cfg.model.get("num_classes", 0)
    shape = (args.num_samples, cfg.model.get("input_dim", 2))
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
        if hasattr(process, "train_step"):
            process.train_step(model, x0, optimizer, y=y, grad_clip=grad_clip)
        else:
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
            y_sample = None
            if num_classes > 0:
                y_sample = torch.randint(0, num_classes, (args.num_samples,), device=device)

            samples = sample(
                model=model,
                process=process,
                shape=shape,
                device=device,
                class_labels=y_sample,
                use_tqdm=False,
            )

        # Sequential frame numbering for ffmpeg compatibility
        save_path = img_dir / f"frame_{frame_idx:07d}.png"
        visualize_distribution(
            samples=samples,
            save_path=save_path,
            title=f"{paradigm_type} at iters: {step}",
            labels=y_sample,
            data_min=data_min,
            data_range=data_range,
            num_classes=num_classes,
            minimal=args.minimal,
        )
        frame_idx += 1
        model.train()

    # Hold last frame for extra seconds by duplicating it
    if args.hold_last_s > 0 and frame_idx > 0:
        last_frame = img_dir / f"frame_{frame_idx - 1:07d}.png"
        extra_frames = int(args.fps * args.hold_last_s)
        for i in range(extra_frames):
            shutil.copy2(last_frame, img_dir / f"frame_{frame_idx + i:07d}.png")
        frame_idx += extra_frames
        print(f"Duplicated last frame {extra_frames}x for {args.hold_last_s}s hold.")

    # Stitch frames into video
    print(f"Stitching {frame_idx} frames into video at {args.fps} fps ...")
    images_to_video(img_dir, video_path, fps=args.fps, filename_pattern="frame_%07d.png")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
