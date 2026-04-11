"""
Unified dataloader creation from config.

This module provides a single entry point for creating dataloaders
based on configuration, routing to the appropriate dataset implementation.
"""
from torch.utils.data import DataLoader

from .datasets import build_synthetic2d_dataloader


def _get(cfg, *keys, default=None):
    """Get nested value from config (works with both dict and OmegaConf)."""
    value = cfg
    for key in keys:
        if hasattr(value, "get"):
            value = value.get(key)
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
        if value is None:
            return default
    return value


def create_dataloader(cfg, world_size: int = 1) -> DataLoader:
    """Create dataloader from config.

    Args:
        cfg: Config object (OmegaConf or dict) with 'data' section.
        world_size: Unused, kept for API compatibility.

    Returns:
        DataLoader for the synthetic2d dataset.
    """
    seed = _get(cfg, "training", "seed", default=0)

    return build_synthetic2d_dataloader(
        distribution=_get(cfg, "data", "distribution", default="gaussian_mixture"),
        num_samples=_get(cfg, "data", "num_samples", default=100000),
        num_classes=_get(cfg, "model", "num_classes", default=8),
        batch_size=_get(cfg, "data", "batch_size"),
        num_workers=_get(cfg, "data", "num_workers", default=0),
        seed=seed,
    )
