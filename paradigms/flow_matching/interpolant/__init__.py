"""
Interpolants (probability paths) for flow matching.

Available interpolants:
- Interpolant: Abstract base class
- LinearInterpolant: x_t = (1-t)x_0 + t*x_1 (optimal transport path)
"""

from .base import Interpolant
from .linear import LinearInterpolant

__all__ = [
    "Interpolant",
    "LinearInterpolant",
]
