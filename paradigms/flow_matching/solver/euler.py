"""
Euler ODE solver for flow matching.

First-order explicit Euler method for solving:
    dx/dt = v_θ(x, t)
"""
import torch
import torch.nn as nn

from .base import ODESolver


class EulerSolver(ODESolver):
    """First-order Euler method.

    Update rule:
        x_{t+dt} = x_t + dt * v_θ(x_t, t)

    For sampling (t: 1 → 0), dt is negative.

    This is the simplest ODE solver. Fast but may require more steps
    for accuracy compared to higher-order methods.
    """

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Perform one Euler step.

        Args:
            model: Velocity model v_θ(x, t).
            x: Current state x_t, shape (B, ...).
            t: Current time, shape (B,).
            dt: Time step (negative for t: 1→0), shape (B,).
            **kwargs: Additional model arguments.

        Returns:
            Next state x_{t+dt}.
        """
        # v_θ(x_t, t)
        v = model(x, t, **kwargs)

        # Reshape dt for broadcasting
        dt = self._reshape_dt(dt, x)

        # x_{t+dt} = x_t + dt * v_θ(x_t, t)
        return x + dt * v

    def _reshape_dt(self, dt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape dt for broadcasting with x."""
        while dt.dim() < x.dim():
            dt = dt.unsqueeze(-1)
        return dt

