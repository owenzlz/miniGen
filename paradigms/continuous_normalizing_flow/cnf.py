"""
Continuous Normalizing Flow (CNF / FFJORD) generative process.

Reference: Grathwohl et al., "FFJORD: Free-form Continuous Dynamics
for Scalable Reversible Generative Models", ICLR 2019.

Training: Maximize log-likelihood via the instantaneous change of variables:
    log p(x_0) = log p(x_T) + integral_0^T Tr(dv/dx) dt

The velocity field v_theta(x, t) is parameterized by a neural network.
For 2D data, the exact trace of the Jacobian is computed efficiently.

Sampling: Solve ODE dx/dt = v(x, t) from t=1 (noise) to t=0 (data).
"""
import math
from typing import Optional

import torch
import torch.nn as nn


def _exact_trace_2d(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute exact trace of Jacobian dv/dx for 2D data.

    Args:
        v: Velocity field output (B, 2). Must be in the computation graph of x.
        x: Input positions (B, 2). Must have requires_grad=True.

    Returns:
        Trace (B,).
    """
    dim = x.shape[1]
    trace = torch.zeros(x.shape[0], device=x.device)
    for d in range(dim):
        grad_d = torch.autograd.grad(
            v[:, d].sum(), x, create_graph=True, retain_graph=True
        )[0]
        trace = trace + grad_d[:, d]
    return trace


class ContinuousNormalizingFlow:
    """Continuous Normalizing Flow (FFJORD) generative process.

    During training, integrates the ODE forward (data -> noise) while
    accumulating the log-determinant via the trace of the Jacobian.
    Uses Euler integration with a configurable number of steps.

    Args:
        num_integration_steps: Number of Euler steps for ODE integration
            during training.
    """

    def __init__(self, num_integration_steps: int = 10):
        self.num_integration_steps = num_integration_steps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (CNF manages time internally)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Negative log-likelihood via continuous change of variables.

        Integrates ODE from t=0 (data) to t=1 (latent) using Euler method,
        accumulating the trace of the Jacobian for the log-determinant.

        NLL = -log p_prior(x_T) - integral_0^1 Tr(dv/dx) dt

        Args:
            model: Velocity network v_theta(x, t).
            x0: Data samples (B, D).
            t: Ignored.
        """
        batch_size, dim = x0.shape
        device = x0.device
        dt = 1.0 / self.num_integration_steps

        # Start from data, integrate forward to latent
        x = x0.detach().requires_grad_(True)
        log_det = torch.zeros(batch_size, device=device)

        for i in range(self.num_integration_steps):
            t_val = i * dt
            t_batch = torch.full((batch_size,), t_val, device=device)

            v = model(x, t_batch, **kwargs)

            # Exact trace for 2D (or small dim)
            trace = _exact_trace_2d(v, x)

            log_det = log_det + trace * dt
            x = x + v * dt

            # Re-enable grad for next step (x is now non-leaf but requires_grad)
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

        # x is now approximately x_T (should be close to standard normal)
        # NLL = -log N(x_T; 0, I) - log_det
        log_prior = -0.5 * (x ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        nll = -log_prior - log_det

        return nll.mean()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        num_steps: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples by solving ODE from t=1 (noise) to t=0 (data).

        Uses Euler integration: x_{t+dt} = x_t + dt * v(x_t, t)

        Args:
            model: Velocity network v_theta(x, t).
            shape: (num_samples, data_dim).
            device: Device.
            num_steps: Number of Euler steps.

        Returns:
            Generated samples at t=0.
        """
        model.eval()
        batch_size = shape[0]
        dt = 1.0 / num_steps

        # Start from noise at t=1
        x = torch.randn(shape, device=device)

        # Integrate backward: t = 1 -> 0
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_batch = torch.full((batch_size,), t_val, device=device)
            v = model(x, t_batch, **kwargs)
            x = x - v * dt  # Negative direction: t=1 to t=0

        return x
