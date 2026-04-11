"""
Shortcut Models generative process.

Extends flow matching by conditioning the velocity network on step size
(represented as dt_base = log2(num_inference_steps)), enabling flexible
inference-time step budgets with a single network and single training phase.

Training uses two losses in a combined batch:
  1. Base flow matching loss (dt_base=log2(N)): standard velocity regression
     at the finest step resolution.
  2. Self-consistency loss (dt_base < log2(N)): a big step should match the
     composition of two half-steps. The target is stop-gradient.

The dt_base representation:
  - dt_base=0 -> dt=1 (one-step generation)
  - dt_base=1 -> dt=1/2 (two-step generation)
  - dt_base=k -> dt=1/2^k (2^k-step generation)
  - dt_base=log2(N) -> dt=1/N (N-step, base FM resolution)

Time convention (this codebase): t=0 is data, t=1 is noise, sampling goes t=1->0.

Reference:
- Frans et al., "One Step Diffusion via Shortcut Models", 2024.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutModel:
    """Shortcut Models generative process.

    Args:
        num_steps: N, the base number of steps (finest resolution). Must be a power of 2.
        bootstrap_every: Batch split ratio. 1/bootstrap_every of each batch is
            used for self-consistency; the rest is standard flow matching.
    """

    def __init__(
        self,
        num_steps: int = 128,
        bootstrap_every: int = 4,
    ):
        self.num_steps = num_steps
        self.bootstrap_every = bootstrap_every
        self.log2_N = int(math.log2(num_steps))

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return dummy timesteps (shortcut training manages time internally)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute combined base FM + self-consistency training loss.

        Follows the official implementation:
        1. Split batch into bootstrap (self-consistency) and flow matching portions.
        2. Pre-compute bootstrap targets via two half-step model calls (stop-grad).
        3. Merge both portions and run a single forward pass to compute loss.

        Args:
            model: Velocity network v_theta(x, t, d=dt_base, ...).
            x0: Clean data samples (B, D).
            t: Ignored (timestep logic is internal).
            **kwargs: Passed through to model (e.g., y for class labels).

        Returns:
            Scalar loss.
        """
        B = x0.shape[0]
        device = x0.device

        n_bst = B // self.bootstrap_every
        n_flow = B - n_bst

        z = torch.randn_like(x0)

        # ==== Bootstrap (self-consistency) targets ====
        if n_bst > 0:
            # Sample dt_base uniformly from {0, 1, ..., log2_N - 1}
            # dt_base=0 means one-step (biggest shortcut), dt_base=log2_N-1 is next-to-finest
            dt_base_bst = torch.randint(
                0, self.log2_N, (n_bst,), device=device
            ).float()
            dt_bst = 1.0 / (2.0 ** dt_base_bst)  # raw step sizes
            dt_base_target = dt_base_bst + 1  # half-step for target
            dt_half = dt_bst / 2

            # Sample t in [dt, 1] (our convention: t=0 data, t=1 noise)
            # Need t >= dt so that the step of size dt stays within [0, 1]
            t_bst = dt_bst + (1.0 - dt_bst) * torch.rand(n_bst, device=device)

            x0_bst, z_bst = x0[:n_bst], z[:n_bst]
            x_t_bst = (
                (1 - t_bst.unsqueeze(1)) * x0_bst + t_bst.unsqueeze(1) * z_bst
            )

            # Target: compose two half-steps (stop-gradient)
            with torch.no_grad():
                v1 = model(x_t_bst, t_bst, d=dt_base_target, **kwargs)
                x_mid = (x_t_bst - dt_half.unsqueeze(1) * v1).clamp(-4, 4)
                t_mid = t_bst - dt_half
                v2 = model(x_mid, t_mid, d=dt_base_target, **kwargs)
                v_target_bst = ((v1 + v2) / 2).clamp(-4, 4)

        # ==== Flow matching targets ====
        x0_flow, z_flow = x0[n_bst:], z[n_bst:]
        t_flow = torch.rand(n_flow, device=device)
        x_t_flow = (
            (1 - t_flow.unsqueeze(1)) * x0_flow + t_flow.unsqueeze(1) * z_flow
        )
        v_target_flow = z_flow - x0_flow  # standard FM velocity
        dt_base_flow = torch.full(
            (n_flow,), float(self.log2_N), device=device
        )

        # ==== Merge and single forward pass ====
        if n_bst > 0:
            x_t = torch.cat([x_t_bst, x_t_flow], dim=0)
            v_target = torch.cat([v_target_bst, v_target_flow], dim=0)
            t_all = torch.cat([t_bst, t_flow], dim=0)
            dt_all = torch.cat([dt_base_bst, dt_base_flow], dim=0)
        else:
            x_t = x_t_flow
            v_target = v_target_flow
            t_all = t_flow
            dt_all = dt_base_flow

        v_pred = model(x_t, t_all, d=dt_all, **kwargs)
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples by solving the ODE from t=1 to t=0.

        With num_steps=1, uses the learned one-step shortcut (dt_base=0).
        With num_steps=128, uses base FM resolution (dt_base=7).

        Args:
            model: Velocity network v_theta(x, t, d=dt_base, ...).
            shape: (num_samples, data_dim).
            device: Device.
            num_steps: Number of sampling steps (must be a power of 2).
            **kwargs: Passed through to model (e.g., y for class labels).

        Returns:
            Generated samples.
        """
        model.eval()

        x = torch.randn(shape, device=device)
        batch_size = shape[0]
        delta_t = 1.0 / num_steps
        dt_base = math.log2(max(num_steps, 1))
        d_batch = torch.full((batch_size,), dt_base, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * delta_t
            t_batch = torch.full((batch_size,), t_val, device=device)
            v = model(x, t_batch, d=d_batch, **kwargs)
            x = x - delta_t * v

        return x
