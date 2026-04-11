"""
Visualization utilities for training.

Provides functions for visualizing generated samples during training.

These functions only handle visualization - sampling should be done separately.
"""
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def get_target_samples(
    dataloader,
    num_samples: int,
    key: str = "tenPoints",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract target samples and normalization params from dataloader.

    Args:
        dataloader: DataLoader to extract samples from.
        num_samples: Number of samples to collect.
        key: Key for data tensor ("tenPoints" for synthetic).

    Returns:
        Tuple of (samples, labels, data_min, data_range). Any can be None.
    """
    if dataloader is None:
        return None, None, None, None

    dataset = dataloader.dataset

    # Get normalization params if available
    data_min = dataset.data_min.numpy() if hasattr(dataset, 'data_min') else None
    data_range = dataset.data_range.numpy() if hasattr(dataset, 'data_range') else None

    # Collect samples
    sample_list = []
    label_list = []
    for batch in dataloader:
        sample_list.append(batch[key])
        if "intLabel" in batch:
            label_list.append(batch["intLabel"])
        if len(sample_list) * dataloader.batch_size >= num_samples:
            break

    samples = torch.cat(sample_list, dim=0)[:num_samples]
    labels = torch.cat(label_list, dim=0)[:num_samples] if label_list else None

    return samples, labels, data_min, data_range


def compute_axis_limits(data_min=None, data_range=None, padding=0.3):
    """Compute axis limits centered on the data range with padding.

    Args:
        data_min: Min values per dimension, shape (2,).
        data_range: Range per dimension, shape (2,).
        padding: Fractional padding around the data (default 0.3 = 30%).

    Returns:
        (xmin, xmax), (ymin, ymax) tuples.
    """
    if data_min is None or data_range is None:
        return (-4, 4), (-4, 4)
    import numpy as np
    data_min = np.asarray(data_min, dtype=float)
    data_range = np.asarray(data_range, dtype=float)
    center = data_min + data_range / 2
    half_span = float(max(data_range)) / 2 * (1 + padding)
    return (center[0] - half_span, center[0] + half_span), (center[1] - half_span, center[1] + half_span)


def visualize_synthetic2d(
    samples: torch.Tensor,
    save_path: Path,
    step: Optional[int] = None,
    labels: Optional[torch.Tensor] = None,
    target_samples: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None,
    data_min: Optional[torch.Tensor] = None,
    data_range: Optional[torch.Tensor] = None,
    num_classes: int = 0,
) -> Path:
    """Visualize 2D synthetic data as scatter plots.

    Creates side-by-side comparison of target distribution and generated samples.

    Args:
        samples: Generated samples, shape (N, 2), in [-1, 1] normalized space.
        save_path: Path to save the figure.
        step: Optional training step (for title).
        labels: Optional class labels for generated samples.
        target_samples: Optional target distribution samples (in [-1, 1] space).
        target_labels: Optional class labels for target samples.
        data_min: Min values for denormalization.
        data_range: Range values for denormalization.
        num_classes: Number of classes (0 for unconditional).

    Returns:
        Path to saved figure.
    """
    samples = samples.cpu().numpy()

    # Denormalize if params available
    if data_min is not None and data_range is not None:
        samples = (samples + 1) / 2 * data_range + data_min
        if target_samples is not None:
            target_samples = (target_samples + 1) / 2 * data_range + data_min

    # Create side-by-side scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Target distribution
    ax_target = axes[0]
    if target_samples is not None:
        if target_labels is not None and num_classes > 0:
            scatter = ax_target.scatter(
                target_samples[:, 0], target_samples[:, 1],
                c=target_labels, cmap='tab10', s=5, alpha=0.7
            )
            plt.colorbar(scatter, ax=ax_target, label='Class')
        else:
            ax_target.scatter(
                target_samples[:, 0], target_samples[:, 1],
                s=5, alpha=0.7, color='C0'
            )
    xlim, ylim = compute_axis_limits(data_min, data_range)
    ax_target.set_xlim(*xlim)
    ax_target.set_ylim(*ylim)
    ax_target.set_aspect('equal')
    ax_target.set_title('Target Distribution')
    ax_target.grid(True, alpha=0.3)

    # Right: Generated samples
    ax_gen = axes[1]
    if labels is not None:
        labels_np = labels.cpu().numpy()
        scatter = ax_gen.scatter(
            samples[:, 0], samples[:, 1],
            c=labels_np, cmap='tab10', s=5, alpha=0.7
        )
        plt.colorbar(scatter, ax=ax_gen, label='Class')
    else:
        ax_gen.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.7, color='C1')
    ax_gen.set_xlim(*xlim)
    ax_gen.set_ylim(*ylim)
    ax_gen.set_aspect('equal')
    title = f'Generated (Step {step})' if step is not None else 'Generated'
    ax_gen.set_title(title)
    ax_gen.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


def update_visualization_html(log_dir: Path) -> Path:
    """Create or update an HTML file to visualize all saved images.

    Scans the saved_images directory and generates an HTML gallery page
    showing all visualization images sorted by training step.

    Args:
        log_dir: The experiment log directory containing saved_images/.

    Returns:
        Path to the generated HTML file.
    """
    import re

    img_dir = log_dir / "saved_images"
    html_path = log_dir / "visualize.html"

    # Find all PNG images and extract step numbers
    images = []
    if img_dir.exists():
        for img_file in img_dir.glob("*.png"):
            # Extract step number from filename like "samples_0000100.png"
            match = re.search(r'samples_(\d+)\.png', img_file.name)
            if match:
                step = int(match.group(1))
                images.append((step, img_file.name))

    # Sort by step number
    images.sort(key=lambda x: x[0])

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Visualizations - {log_dir.name}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #fff;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .image-card {{
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        .image-card .caption {{
            padding: 12px 15px;
            background: #0f3460;
            font-size: 14px;
            font-weight: 500;
        }}
        .step-badge {{
            display: inline-block;
            background: #e94560;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 8px;
        }}
        .no-images {{
            text-align: center;
            color: #666;
            padding: 50px;
            font-size: 18px;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            cursor: pointer;
        }}
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: #fff;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
        }}
        .refresh-note {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>Training Visualizations</h1>
    <p class="subtitle">{log_dir.name} | {len(images)} images</p>
"""

    if images:
        html_content += '    <div class="gallery">\n'
        for step, img_name in images:
            html_content += f"""        <div class="image-card">
            <img src="saved_images/{img_name}" alt="Step {step}" onclick="openModal(this.src)">
            <div class="caption">
                <span class="step-badge">Step {step:,}</span>
                {img_name}
            </div>
        </div>
"""
        html_content += '    </div>\n'
    else:
        html_content += '    <p class="no-images">No visualization images found yet.</p>\n'

    html_content += """
    <p class="refresh-note">Refresh this page to see new visualizations during training.</p>

    <div id="modal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modal-img" src="">
    </div>

    <script>
        function openModal(src) {
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'block';
        }
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)

    return html_path


def visualize_distribution(
    samples,
    save_path: Path,
    title: str = "",
    labels=None,
    data_min=None,
    data_range=None,
    num_classes: int = 0,
    minimal: bool = False,
) -> Path:
    """Visualize a single 2D distribution as a scatter plot.

    Unlike visualize_synthetic2d (side-by-side), this renders one panel only.

    Args:
        samples: Samples array/tensor of shape (N, 2), in [-1, 1] normalized space.
        save_path: Path to save the figure.
        title: Plot title (e.g., "GAN at iters: 123").
        labels: Optional class labels (tensor or ndarray).
        data_min: Min values for denormalization.
        data_range: Range values for denormalization.
        num_classes: Number of classes (0 for unconditional).
        minimal: If True, strip title, grid, ticks, axes, and colorbar.

    Returns:
        Path to saved figure.
    """
    import numpy as np

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    samples = np.asarray(samples)

    # Denormalize if params available
    if data_min is not None and data_range is not None:
        data_min = np.asarray(data_min)
        data_range = np.asarray(data_range)
        samples = (samples + 1) / 2 * data_range + data_min

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        scatter = ax.scatter(
            samples[:, 0], samples[:, 1],
            c=labels, cmap='tab10', s=5, alpha=0.7,
        )
        if num_classes > 0 and not minimal:
            plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.7, color='C1')

    xlim, ylim = compute_axis_limits(data_min, data_range)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    if minimal:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    else:
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


def images_to_video(
    img_dir: Path,
    output_path: Path,
    fps: int = 120,
    filename_pattern: str = "step_%07d.png",
) -> Path:
    """Stitch sequentially numbered images into an MP4 video using ffmpeg.

    Args:
        img_dir: Directory containing the image frames.
        output_path: Path for the output MP4 video.
        fps: Frames per second.
        filename_pattern: Printf-style filename pattern (e.g., "step_%07d.png").

    Returns:
        Path to the saved video file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(img_dir / filename_pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path
