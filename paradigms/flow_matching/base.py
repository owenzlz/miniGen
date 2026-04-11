"""
Base class for flow matching models.

================================================================================
FLOW MATCHING NOTATION (Lipman et al., 2022; Liu et al., 2022)
================================================================================

Flow matching learns a velocity field v_θ(x, t) that generates a probability path
from noise distribution p_1 to data distribution p_0.

Interpolant (probability path):
    x_t = I(x_0, x_1, t)

Velocity field (target):
    v_t = dI/dt = ∂I/∂t

Training objective:
    L = E_{t, x_0, x_1} ||v_θ(x_t, t) - v_t||²

Sampling (ODE from t=1 to t=0):
    dx/dt = v_θ(x, t)
    x_0 = x_1 + ∫_1^0 v_θ(x_t, t) dt

================================================================================
References:
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Liu et al., "Flow Straight and Fast", ICLR 2023.
- Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants", ICLR 2023.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn

from .interpolant.base import Interpolant
from .solver.base import ODESolver


class FlowMatchingBase(ABC, nn.Module):
    """Abstract base class for flow matching models.

    Subclasses implement specific flow matching variants by defining:
    - forward_process: How to compute x_t and target velocity
    - training_loss: The training objective
    - sample: How to generate samples via ODE solving
    """

    def __init__(
        self,
        interpolant: Interpolant,
        solver: Optional[ODESolver] = None,
    ):
        """Initialize flow matching model.

        Args:
            interpolant: Interpolant defining the probability path.
            solver: ODE solver for sampling (default: Euler).
        """
        super().__init__()
        self.interpolant = interpolant
        self.solver = solver

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Sample random timesteps t ~ U[0, 1] for training.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Timesteps in [0, 1], shape (batch_size,).
        """
        return torch.rand(batch_size, device=device)

    @abstractmethod
    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute x_t and target velocity.

        Args:
            x0: Clean data samples.
            t: Time values in [0, 1].
            noise: Optional pre-sampled noise (x_1).

        Returns:
            Tuple of (x_t, velocity_target, noise).
        """
        pass

    @abstractmethod
    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            model: Velocity prediction model.
            x0: Clean data samples.
            t: Time values.
            **kwargs: Additional model arguments (e.g., class labels).

        Returns:
            Scalar loss.
        """
        pass

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples by solving the ODE.

        Args:
            model: Velocity prediction model.
            shape: Shape of samples to generate.
            device: Device to generate on.
            num_steps: Number of ODE solver steps.
            **kwargs: Additional model arguments.

        Returns:
            Generated samples.
        """
        pass
