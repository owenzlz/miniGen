"""
Base class for interpolants (probability paths).

An interpolant defines how to interpolate between data x_0 and noise x_1,
creating a probability path for flow matching.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""
from abc import ABC, abstractmethod
import torch


class Interpolant(ABC):
    """Abstract base class for interpolants (probability paths).

    An interpolant I(x_0, x_1, t) defines:
    - How to interpolate between data x_0 and noise x_1
    - The target velocity v_t = dI/dt

    Common choices:
    - Linear:  x_t = (1-t)x_0 + t*x_1
    - VP:      x_t = α_t*x_0 + σ_t*x_1  (variance preserving)
    """

    @abstractmethod
    def __call__(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute x_t = I(x_0, x_1, t).

        Args:
            x0: Data samples, shape (B, ...).
            x1: Noise samples, shape (B, ...).
            t: Time values in [0, 1], shape (B,) or (B, 1, ...).

        Returns:
            Interpolated samples x_t, shape (B, ...).
        """
        pass

    @abstractmethod
    def velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute target velocity v_t = dI/dt.

        Args:
            x0: Data samples.
            x1: Noise samples.
            t: Time values.

        Returns:
            Target velocity, shape (B, ...).
        """
        pass
