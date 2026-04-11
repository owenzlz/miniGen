"""
Linear interpolant for flow matching.

This is the optimal transport (OT) path between data and noise distributions.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""
import torch

from .base import Interpolant


class LinearInterpolant(Interpolant):
    """Linear interpolant (optimal transport path).

    Interpolation:
        x_t = (1-t) x_0 + t x_1

    where:
        x_0 ~ p_data  (clean data)
        x_1 ~ N(0, I) (noise)
        t ∈ [0, 1]

    At t=0: x_t = x_0 (data)
    At t=1: x_t = x_1 (noise)

    Velocity (derivative w.r.t. t):
        v_t = dx_t/dt = x_1 - x_0

    This is a constant velocity field along the straight-line path.
    """

    def __call__(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute x_t = (1-t) x_0 + t x_1.

        Args:
            x0: Data samples, shape (B, ...).
            x1: Noise samples, shape (B, ...).
            t: Time values in [0, 1], shape (B,).

        Returns:
            Interpolated samples x_t.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, ...) or (B, 1)
        t = self._reshape_t(t, x0)
        return (1 - t) * x0 + t * x1

    def velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute target velocity v_t = x_1 - x_0.

        For linear interpolation, the velocity is constant (independent of t).

        Args:
            x0: Data samples.
            x1: Noise samples.
            t: Time values (unused for linear interpolant).

        Returns:
            Target velocity v_t = x_1 - x_0.
        """
        return x1 - x0

    def _reshape_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape t for broadcasting with x.

        Args:
            t: Time tensor, shape (B,).
            x: Data tensor, shape (B, ...).

        Returns:
            Reshaped t with shape (B, 1, 1, ...).
        """
        # Add dimensions to match x: (B,) -> (B, 1, 1, ...)
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        return t
