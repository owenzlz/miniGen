"""
Improved Consistency Training (iCT) generative process.

Reference: Song & Dhariwal, "Improved Techniques for Training Consistency Models",
ICLR 2024 Oral.

Trains a consistency model from scratch (no distillation) to map any point on a
noise trajectory directly to the clean data. Key tricks from the paper:
  1. No EMA teacher (stopgrad only)
  2. Pseudo-Huber loss
  3. Lognormal timestep sampling
  4. Progressive N schedule (discretization doubling)
  5. Karras sigma discretization
  6. EDM preconditioning (c_skip, c_out, c_in)
  7. Inverse step-size loss weighting
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class ConsistencyTraining:
    """Improved Consistency Training (iCT) generative process.

    Maps any noisy input directly to the clean data in a single step,
    trained via a self-consistency objective with progressive schedule.

    Args:
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        sigma_data: Data standard deviation (for EDM preconditioning).
        rho: Karras sigma schedule exponent.
        initial_timesteps: Starting number of discretization steps (s0).
        final_timesteps: Final number of discretization steps (s1).
        total_training_steps: Total training iterations (K) for progressive schedule.
        p_mean: Mean of lognormal timestep sampling distribution.
        p_std: Std of lognormal timestep sampling distribution.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        initial_timesteps: int = 10,
        final_timesteps: int = 150,
        total_training_steps: int = 20000,
        p_mean: float = -0.4,
        p_std: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.initial_timesteps = initial_timesteps  # s0
        self.final_timesteps = final_timesteps      # s1
        self.total_training_steps = total_training_steps  # K
        self.p_mean = p_mean
        self.p_std = p_std

        self._step = 0

    def _current_num_timesteps(self) -> int:
        """Progressive schedule N(k) from Eq. 18 in the paper.

        N(k) = ceil(sqrt((k/K)*((s1+1)^2 - s0^2) + s0^2) - 1) + 1
        """
        k = self._step
        K = self.total_training_steps
        s0 = self.initial_timesteps
        s1 = self.final_timesteps

        ratio = k / max(K, 1)
        N = math.ceil(math.sqrt(ratio * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1
        return max(N, 2)  # Need at least 2 for a pair

    def _karras_sigmas(self, N: int, device: torch.device) -> torch.Tensor:
        """Build N Karras sigma values from sigma_min to sigma_max.

        sigma_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho
        """
        rho_inv = 1.0 / self.rho
        min_inv = self.sigma_min ** rho_inv
        max_inv = self.sigma_max ** rho_inv

        indices = torch.arange(N, device=device, dtype=torch.float32)
        sigmas = (min_inv + indices / (N - 1) * (max_inv - min_inv)) ** self.rho
        return sigmas

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (consistency training manages time internally)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute iCT training loss.

        Args:
            model: Network F_theta(x, sigma). Receives sigma as the t argument.
            x0: Clean data samples (B, D).
            t: Ignored (timestep logic is internal).
        """
        batch_size = x0.shape[0]
        device = x0.device

        # 1. Progressive schedule: compute current N
        N = self._current_num_timesteps()

        # 2. Karras sigma discretization
        sigmas = self._karras_sigmas(N, device)  # (N,)

        # 3. Lognormal-weighted index sampling over [0, N-2]
        # Sample from lognormal, then convert to categorical weights
        num_pairs = N - 1
        indices = torch.arange(num_pairs, device=device, dtype=torch.float32)
        # Use midpoint sigma for lognormal weighting
        mid_sigmas = (sigmas[:-1] + sigmas[1:]) / 2
        log_weights = -((torch.log(mid_sigmas) - self.p_mean) ** 2) / (2 * self.p_std ** 2)
        weights = torch.softmax(log_weights, dim=0)

        # Sample indices
        n = torch.multinomial(weights.expand(batch_size, -1), num_samples=1).squeeze(1)  # (B,)

        sigma_n = sigmas[n]          # lower sigma
        sigma_n1 = sigmas[n + 1]     # higher sigma

        # 4. Create noisy pairs with same noise, different scale
        z = torch.randn_like(x0)
        x_n1 = x0 + sigma_n1.unsqueeze(1) * z   # noisier
        x_n = x0 + sigma_n.unsqueeze(1) * z      # less noisy

        # 5. Online prediction (with gradient)
        online = self._consistency_fn(model, x_n1, sigma_n1, **kwargs)

        # 6. Target prediction (stopgrad — same model, no grad)
        with torch.no_grad():
            target = self._consistency_fn(model, x_n, sigma_n, **kwargs)

        # 7. Loss with inverse step-size weighting
        lam = 1.0 / (sigma_n1 - sigma_n)  # (B,)
        loss_per_sample = self._pseudo_huber_loss(online, target)  # (B,)
        loss = (lam * loss_per_sample).mean()

        self._step += 1
        return loss

    def _consistency_fn(
        self, model: nn.Module, x: torch.Tensor, sigma: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Apply EDM-style skip connection parameterization.

        f(x, sigma) = c_skip * x + c_out * F_theta(c_in * x, sigma)

        where:
            c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
            c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
            c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
        """
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2

        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * self.sigma_data / torch.sqrt(s2 + sd2)
        c_in = 1.0 / torch.sqrt(s2 + sd2)

        # Reshape for broadcasting: (B,) -> (B, 1)
        c_skip = c_skip.unsqueeze(1)
        c_out = c_out.unsqueeze(1)
        c_in = c_in.unsqueeze(1)

        # Pass sigma as the time input to the model
        model_out = model(c_in * x, sigma, **kwargs)
        return c_skip * x + c_out * model_out

    def _pseudo_huber_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pseudo-Huber loss: sqrt(||x-y||^2 + c^2) - c.

        c = 0.00054 * sqrt(d) where d is the data dimension.
        Returns per-sample loss (B,).
        """
        d = x.shape[1]
        c = 0.00054 * math.sqrt(d)
        diff_sq = ((x - y) ** 2).sum(dim=1)
        return torch.sqrt(diff_sq + c ** 2) - c

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        num_steps: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples via consistency model.

        Single-step: sample noise at sigma_max, apply consistency function once.
        Multi-step: iteratively denoise and re-noise at decreasing sigma levels.

        Args:
            model: Consistency model.
            shape: (num_samples, data_dim).
            device: Device.
            num_steps: 1 for single-step, >1 for multi-step refinement.

        Returns:
            Generated samples.
        """
        model.eval()
        batch_size = shape[0]

        # Start from pure noise at sigma_max
        x = torch.randn(shape, device=device) * self.sigma_max
        sigma = torch.full((batch_size,), self.sigma_max, device=device)

        # Single-step: just apply consistency function
        x = self._consistency_fn(model, x, sigma, **kwargs)

        if num_steps > 1:
            # Multi-step: build Karras sigmas for refinement
            sigmas = self._karras_sigmas(num_steps + 1, device)  # includes sigma_min

            # Iterate from second-highest sigma down to sigma_min
            for i in range(len(sigmas) - 2, 0, -1):
                sigma_i = sigmas[i]
                # Re-noise: add noise at sigma_i
                z = torch.randn_like(x)
                x = x + torch.sqrt(sigma_i ** 2 - self.sigma_min ** 2) * z
                # Denoise via consistency function
                sigma_batch = torch.full((batch_size,), sigma_i.item(), device=device)
                x = self._consistency_fn(model, x, sigma_batch, **kwargs)

        return x
