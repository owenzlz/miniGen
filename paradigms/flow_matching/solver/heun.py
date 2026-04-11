"""
Heun ODE solver for flow matching.

Second-order Heun method (improved Euler / explicit trapezoidal) for solving:
    dx/dt = v_θ(x, t)
"""
import torch
import torch.nn as nn

from .base import ODESolver


class HeunSolver(ODESolver):
    """Second-order Heun method (improved Euler).

    Update rule:
        x̃ = x_t + dt * v_θ(x_t, t)                    # Euler predictor
        x_{t+dt} = x_t + dt/2 * (v_θ(x_t, t) + v_θ(x̃, t+dt))  # Trapezoidal corrector

    This is more accurate than Euler with the same step size,
    but requires 2 model evaluations per step.

    Also known as:
    - Improved Euler method
    - Explicit trapezoidal method
    - RK2 (a form of 2nd-order Runge-Kutta)
    """

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Perform one Heun step.

        Args:
            model: Velocity model v_θ(x, t).
            x: Current state x_t, shape (B, ...).
            t: Current time, shape (B,).
            dt: Time step (negative for t: 1→0), shape (B,).
            **kwargs: Additional model arguments.

        Returns:
            Next state x_{t+dt}.
        """
        # Reshape dt for broadcasting
        dt_broadcast = self._reshape_dt(dt, x)

        # k1 = v_θ(x_t, t)
        k1 = model(x, t, **kwargs)

        # Euler predictor: x̃ = x_t + dt * k1
        x_pred = x + dt_broadcast * k1

        # k2 = v_θ(x̃, t + dt)
        t_next = t + dt
        k2 = model(x_pred, t_next, **kwargs)

        # Trapezoidal corrector: x_{t+dt} = x_t + dt/2 * (k1 + k2)
        return x + dt_broadcast * 0.5 * (k1 + k2)

    def _reshape_dt(self, dt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape dt for broadcasting with x."""
        while dt.dim() < x.dim():
            dt = dt.unsqueeze(-1)
        return dt
