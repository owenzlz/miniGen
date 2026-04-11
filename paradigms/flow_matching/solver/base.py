"""
Base class for ODE solvers.

ODE solvers integrate the flow ODE: dx/dt = v_θ(x, t) from t=1 to t=0
to generate samples from the learned distribution.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ODESolver(ABC):
    """Abstract base class for ODE solvers.

    Solves the ODE: dx/dt = v_θ(x, t) from t=1 to t=0.

    Subclasses implement different integration methods:
    - Euler: First-order, 1 function evaluation per step
    - Heun: Second-order, 2 function evaluations per step
    - RK4: Fourth-order, 4 function evaluations per step
    """

    @abstractmethod
    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Perform one integration step.

        Args:
            model: Velocity model v_θ.
            x: Current state x_t.
            t: Current time.
            dt: Time step (negative for backward integration).
            **kwargs: Additional model arguments.

        Returns:
            Next state x_{t+dt}.
        """
        pass

    @torch.no_grad()
    def solve(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 0.0,
        num_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Solve ODE from t_start to t_end.

        Args:
            model: Velocity model v_θ.
            x: Initial state at t_start.
            t_start: Start time (default 1.0 for sampling).
            t_end: End time (default 0.0 for sampling).
            num_steps: Number of integration steps.
            **kwargs: Additional model arguments.

        Returns:
            Final state at t_end.
        """
        model.eval()
        device = x.device
        batch_size = x.shape[0]

        dt = (t_end - t_start) / num_steps
        t = t_start

        for _ in range(num_steps):
            t_batch = torch.full((batch_size,), t, device=device)
            dt_batch = torch.full((batch_size,), dt, device=device)
            x = self.step(model, x, t_batch, dt_batch, **kwargs)
            t = t + dt

        return x
