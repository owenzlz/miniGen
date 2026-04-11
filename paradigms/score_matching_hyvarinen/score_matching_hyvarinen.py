"""
Explicit Score Matching (Hyvarinen, 2005).

Reference: Hyvarinen, "Estimation of Non-Normalized Statistical Models
by Score Matching", JMLR 2005.

Training: Minimize E[Tr(dS/dx) + 1/2 ||S(x)||^2] where S = s_theta(x).
    This is equivalent to E[||s_theta(x) - nabla_x log p(x)||^2] but
    avoids computing the unknown true score via integration by parts.

Sampling: Langevin dynamics using the learned score.
"""
from typing import Optional

import torch
import torch.nn as nn


def _jacobian_trace_2d(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute exact trace of Jacobian ds/dx for 2D data.

    Tr(ds/dx) = ds_0/dx_0 + ds_1/dx_1

    Args:
        s: Score output (B, 2). Must be in the computation graph of x.
        x: Input positions (B, 2). Must have requires_grad=True.

    Returns:
        Trace (B,).
    """
    trace = torch.zeros(x.shape[0], device=x.device)
    for d in range(x.shape[1]):
        grad_d = torch.autograd.grad(
            s[:, d].sum(), x, create_graph=True, retain_graph=True
        )[0]
        trace = trace + grad_d[:, d]
    return trace


class ScoreMatchingHyvarinen:
    """Explicit Score Matching (Hyvarinen, 2005).

    The model directly predicts the score s_theta(x) = nabla_x log p(x)
    on clean data (no noise added). The loss uses Hyvarinen's trick
    (integration by parts) to avoid needing the true score:

        L = E_x[Tr(ds_theta/dx) + 1/2 ||s_theta(x)||^2]

    Sampling uses Langevin dynamics with the learned score.

    Args:
        langevin_steps: Number of Langevin steps for sampling.
        langevin_eps: Step size for Langevin dynamics.
    """

    def __init__(
        self,
        langevin_steps: int = 1000,
        langevin_eps: float = 1e-4,
    ):
        self.langevin_steps = langevin_steps
        self.langevin_eps = langevin_eps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (explicit score matching doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Hyvarinen's explicit score matching loss.

        L = E_x[Tr(ds_theta/dx) + 1/2 ||s_theta(x)||^2]

        This is derived from ||s_theta(x) - nabla_x log p(x)||^2 via
        integration by parts, eliminating the unknown true score.

        Args:
            model: Score network s_theta(x, t) -> score estimate (B, D).
            x0: Clean data samples (B, D). No noise added.
            t: Ignored.
        """
        x = x0.detach().requires_grad_(True)

        # Model predicts the score directly
        score = model(x, t, **kwargs)

        # Term 1: Tr(ds_theta/dx) — Jacobian trace via autograd
        trace = _jacobian_trace_2d(score, x)

        # Term 2: 1/2 ||s_theta(x)||^2
        score_norm_sq = 0.5 * (score ** 2).sum(dim=-1)

        # Hyvarinen loss: E[Tr + 1/2 ||s||^2]
        loss = (trace + score_norm_sq).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples via Langevin dynamics.

        x_{k+1} = x_k + (eps/2) * s_theta(x_k) + sqrt(eps) * z_k

        Args:
            model: Score network.
            shape: (num_samples, data_dim).
            device: Device.
            num_steps: Total Langevin steps (overrides default).
        """
        model.eval()
        batch_size = shape[0]
        steps = num_steps if num_steps else self.langevin_steps
        t_batch = torch.zeros(batch_size, device=device)

        # Initialize from broad noise
        x = torch.randn(shape, device=device)

        for _ in range(steps):
            score = model(x, t_batch, **kwargs)
            z = torch.randn_like(x)
            x = x + (self.langevin_eps / 2) * score + torch.sqrt(
                torch.tensor(self.langevin_eps, device=device)
            ) * z

        return x
