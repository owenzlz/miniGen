"""
Core sampling utilities for generating samples from diffusion and flow matching models.

This module provides the unified sampling interface used by both training
visualization and standalone inference.
"""
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def sample(
    model: nn.Module,
    process,
    shape: Tuple[int, ...],
    device: torch.device,
    sampler=None,
    num_steps: Optional[int] = None,
    class_labels: Optional[Union[int, List[int], torch.Tensor]] = None,
    cfg_scale: float = 1.0,
    cfg_wrapper_class=None,
    num_classes: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_tqdm: bool = True,
) -> torch.Tensor:
    """Generate samples from the model.

    Unified sampling function that works with both diffusion (DDPM/DDIM) and
    flow matching models. Supports classifier-free guidance and batched generation.

    Args:
        model: The denoising/velocity model.
        process: The generative process (DDPM or FlowMatching).
        shape: Full shape of samples to generate (N, C, H, W) or (N, D).
        device: Device to generate on.
        sampler: Optional diffusion sampler (DDPMSampler or DDIMSampler).
            If None, uses process.sample() directly.
        num_steps: Number of sampling steps. For DDIM, enables accelerated sampling.
            For flow matching, controls ODE integration steps.
        class_labels: Optional class labels. Can be:
            - int: Same label for all samples
            - List[int]: Label per sample
            - Tensor: Label tensor
            - None: Unconditional generation
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
        cfg_wrapper_class: CFG wrapper class (e.g., CFGWrapper). Required if cfg_scale > 1.
        num_classes: Number of classes (required for CFG).
        batch_size: If provided, generates in batches to manage memory.
            If None, generates all samples at once.
        use_tqdm: Show progress bar (only when batching).

    Returns:
        Tensor of generated samples with shape `shape`.
    """
    model.eval()
    num_samples = shape[0]
    sample_shape = shape[1:]  # Shape without batch dimension

    # Handle batched generation
    if batch_size is not None and batch_size < num_samples:
        return _sample_batched(
            model=model,
            process=process,
            num_samples=num_samples,
            sample_shape=sample_shape,
            device=device,
            sampler=sampler,
            num_steps=num_steps,
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            cfg_wrapper_class=cfg_wrapper_class,
            num_classes=num_classes,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
        )

    # Prepare class labels
    y = _prepare_labels(class_labels, num_samples, device)

    # Wrap model with CFG if needed
    sample_model = model
    if cfg_scale > 1.0 and y is not None:
        if cfg_wrapper_class is None:
            raise ValueError("cfg_wrapper_class required when cfg_scale > 1.0")
        if num_classes is None:
            raise ValueError("num_classes required when cfg_scale > 1.0")
        sample_model = cfg_wrapper_class(model, cfg_scale, null_class=num_classes)

    # Build kwargs
    kwargs = {"y": y}
    if num_steps is not None:
        kwargs["num_steps"] = num_steps

    # Sample using sampler or process
    if sampler is not None:
        samples = sampler.sample(sample_model, shape, device, **kwargs)
    else:
        samples = process.sample(sample_model, shape, device, **kwargs)

    return samples


def _prepare_labels(
    class_labels: Optional[Union[int, List[int], torch.Tensor]],
    num_samples: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Prepare class labels tensor from various input formats."""
    if class_labels is None:
        return None

    if isinstance(class_labels, int):
        return torch.full((num_samples,), class_labels, device=device, dtype=torch.long)
    elif isinstance(class_labels, torch.Tensor):
        return class_labels.to(device)
    else:
        # List or other iterable
        return torch.tensor(class_labels[:num_samples], device=device, dtype=torch.long)


def _sample_batched(
    model: nn.Module,
    process,
    num_samples: int,
    sample_shape: Tuple[int, ...],
    device: torch.device,
    sampler=None,
    num_steps: Optional[int] = None,
    class_labels: Optional[Union[int, List[int], torch.Tensor]] = None,
    cfg_scale: float = 1.0,
    cfg_wrapper_class=None,
    num_classes: Optional[int] = None,
    batch_size: int = 16,
    use_tqdm: bool = True,
) -> torch.Tensor:
    """Generate samples in batches to manage memory."""
    samples_list = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    iterator = range(num_batches)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Sampling")

    for i in iterator:
        current_batch = min(batch_size, num_samples - i * batch_size)
        batch_shape = (current_batch,) + sample_shape

        # Get batch labels
        batch_labels = None
        if class_labels is not None:
            if isinstance(class_labels, int):
                batch_labels = class_labels
            elif isinstance(class_labels, torch.Tensor):
                start_idx = i * batch_size
                batch_labels = class_labels[start_idx:start_idx + current_batch]
            else:
                start_idx = i * batch_size
                batch_labels = class_labels[start_idx:start_idx + current_batch]

        batch_samples = sample(
            model=model,
            process=process,
            shape=batch_shape,
            device=device,
            sampler=sampler,
            num_steps=num_steps,
            class_labels=batch_labels,
            cfg_scale=cfg_scale,
            cfg_wrapper_class=cfg_wrapper_class,
            num_classes=num_classes,
            batch_size=None,  # Don't recurse into batching
            use_tqdm=False,
        )
        samples_list.append(batch_samples)

    return torch.cat(samples_list, dim=0)[:num_samples]
