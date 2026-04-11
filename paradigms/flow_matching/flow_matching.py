"""
Flow Matching implementation.

Reference:
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Liu et al., "Flow Straight and Fast", ICLR 2023.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FlowMatchingBase, Interpolant, ODESolver
from .interpolant import LinearInterpolant
from .solver import EulerSolver


class FlowMatching(FlowMatchingBase):
    """Flow Matching generative model.

    Learns a velocity field v_θ(x, t) that transports samples from
    noise distribution (t=1) to data distribution (t=0).

    Training:
        1. Sample x_0 ~ p_data, x_1 ~ N(0, I), t ~ U[0, 1]
        2. Compute x_t = interpolant(x_0, x_1, t)
        3. Compute target v_t = d/dt interpolant(x_0, x_1, t)
        4. Minimize ||v_θ(x_t, t) - v_t||²

    Sampling:
        1. Sample x_1 ~ N(0, I)
        2. Solve ODE: dx/dt = v_θ(x, t) from t=1 to t=0
        3. Return x_0
    """

    def __init__(
        self,
        interpolant: Optional[Interpolant] = None,
        solver: Optional[ODESolver] = None,
    ):
        """Initialize flow matching model.

        Args:
            interpolant: Interpolant for probability path (default: LinearInterpolant).
            solver: ODE solver for sampling (default: EulerSolver).
        """
        if interpolant is None:
            interpolant = LinearInterpolant()
        if solver is None:
            solver = EulerSolver()

        super().__init__(interpolant, solver)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, 1].

        For flow matching, we use continuous time t ∈ [0, 1] instead of
        discrete timesteps.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1].
        """
        return torch.rand(batch_size, device=device)

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interpolated sample and target velocity.

        Args:
            x0: Clean data samples, shape (B, ...).
            t: Time values in [0, 1], shape (B,).
            noise: Optional pre-sampled noise x_1 (default: sample from N(0, I)).

        Returns:
            Tuple of (x_t, velocity_target, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # x_t = I(x_0, x_1, t)
        xt = self.interpolant(x0, noise, t)

        # v_t = dI/dt
        velocity = self.interpolant.velocity(x0, noise, t)

        return xt, velocity, noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute flow matching training loss.

        L = E_{t, x_0, x_1} ||v_θ(x_t, t) - v_t||²

        Args:
            model: Velocity prediction model v_θ.
            x0: Clean data samples.
            t: Time values in [0, 1].
            **kwargs: Additional model arguments (e.g., y for class labels).

        Returns:
            Scalar MSE loss.
        """
        # Get x_t and target velocity
        xt, velocity_target, _ = self.forward_process(x0, t)

        # Predict velocity
        velocity_pred = model(xt, t, **kwargs)

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples by solving the ODE from t=1 to t=0.

        Args:
            model: Velocity prediction model v_θ.
            shape: Shape of samples to generate (batch_size, ...).
            device: Device to generate on.
            num_steps: Number of ODE solver steps.
            **kwargs: Additional model arguments.

        Returns:
            Generated samples at t=0.
        """
        model.eval()

        # x_1 ~ N(0, I)
        x = torch.randn(shape, device=device)

        # Solve ODE: dx/dt = v_θ(x, t) from t=1 to t=0
        x = self.solver.solve(
            model, x,
            t_start=1.0,
            t_end=0.0,
            num_steps=num_steps,
            **kwargs
        )

        return x
