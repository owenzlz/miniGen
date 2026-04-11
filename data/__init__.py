"""
Data loading utilities.

Main entry point:
    create_dataloader(cfg) - Create dataloader from config
"""

from .dataloader import create_dataloader
from .datasets import (
    Synthetic2DDataset,
    build_synthetic2d_dataloader,
)

__all__ = [
    "create_dataloader",
    "Synthetic2DDataset",
    "build_synthetic2d_dataloader",
]
