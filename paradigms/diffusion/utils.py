"""
Shared utilities for diffusion models.
"""
from typing import Tuple
import torch


def extract(arr: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from arr at indices t and reshape for broadcasting.

    Args:
        arr: 1D tensor of values indexed by timestep.
        t: Batch of timestep indices, shape (batch_size,).
        x_shape: Shape of tensor to broadcast against.

    Returns:
        Extracted values reshaped to (batch_size, 1, 1, ...).
    """
    batch_size = t.shape[0]
    out = arr.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
