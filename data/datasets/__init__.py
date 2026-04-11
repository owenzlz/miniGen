"""
Dataset implementations.

Each dataset module contains:
- A Dataset class with consistent interface
- A build_xxx_dataloader() function for creating DataLoaders
"""

from .synthetic2d import Synthetic2DDataset, build_synthetic2d_dataloader

__all__ = [
    "Synthetic2DDataset",
    "build_synthetic2d_dataloader",
]
