"""
Score Matching (NCSN) generative process.

Reference: Song & Ermon, "Generative Modeling by Estimating Gradients
of the Data Distribution", NeurIPS 2019.

Training: Denoising score matching with geometric noise levels.
Sampling: Annealed Langevin dynamics.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreMatchingNCSN:
    """Score Matching NCSN (Song & Ermon, 2019) generative process.

    Uses a geometric sequence of noise levels sigma_0 > sigma_1 > ... > sigma_{L-1}.
    The model is trained via denoising score matching to predict the noise
    (equivalent to predicting sigma * score).

    Sampling uses annealed Langevin dynamics, iterating from high to low noise.

    Args:
        num_noise_levels: Number of noise levels L.
        sigma_min: Smallest noise level.
        sigma_max: Largest noise level.
        langevin_steps: Number of Langevin steps per noise level during sampling.
        langevin_eps: Base step size for Langevin dynamics.
    """

    def __init__(
        self,
        num_noise_levels: int = 10,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        langevin_steps: int = 100,
        langevin_eps: float = 1e-5,
    ):
        self.num_noise_levels = num_noise_levels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.langevin_steps = langevin_steps
        self.langevin_eps = langevin_eps

        # Geometric noise schedule: sigma_0 = sigma_max, sigma_{L-1} = sigma_min
        self.sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_max), math.log(sigma_min), num_noise_levels
            )
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random noise level indices.

        Returns:
            Tensor of shape (batch_size,) with integer indices in [0, L-1].
        """
        return torch.randint(0, self.num_noise_levels, (batch_size,), device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Denoising score matching loss.

        The model predicts the noise epsilon added to the data.
        This is equivalent to predicting sigma * score, since:
            score = -epsilon / sigma
            model_output ≈ epsilon

        Loss = E[||model(x + sigma*eps, t) - eps||^2]

        Args:
            model: Score network (predicts noise).
            x0: Clean data (B, D).
            t: Noise level indices (B,).
        """
        sigmas = self.sigmas.to(x0.device)
        sigma = sigmas[t]  # (B,)

        # Perturb data
        noise = torch.randn_like(x0)
        x_noisy = x0 + sigma.unsqueeze(-1) * noise

        # Model predicts noise
        model_out = model(x_noisy, t, **kwargs)

        # MSE loss (with per-level weighting 1/sigma^2 absorbed into the target)
        loss = F.mse_loss(model_out, noise)
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
        """Generate samples via annealed Langevin dynamics.

        For each noise level from high to low:
            For T steps:
                step_size = eps * (sigma_i / sigma_min)^2
                x = x + step_size/2 * score + sqrt(step_size) * z
                  = x - step_size/2 * model(x, i) / sigma_i + sqrt(step_size) * z

        Args:
            model: Score network.
            shape: Sample shape (num_samples, data_dim).
            device: Device.
            num_steps: Total Langevin steps (split across noise levels).
        """
        model.eval()
        sigmas = self.sigmas.to(device)
        batch_size = shape[0]

        steps_per_level = num_steps // self.num_noise_levels if num_steps else self.langevin_steps

        # Initialize from large noise
        x = torch.randn(shape, device=device) * sigmas[0]

        for i in range(self.num_noise_levels):
            sigma = sigmas[i]
            t_batch = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Step size scales with noise level
            alpha = self.langevin_eps * (sigma / sigmas[-1]) ** 2

            for _ in range(steps_per_level):
                # Model predicts noise; score = -noise / sigma
                noise_pred = model(x, t_batch, **kwargs)
                score = -noise_pred / sigma

                # Langevin update
                z = torch.randn_like(x)
                x = x + alpha / 2 * score + torch.sqrt(alpha) * z

        return x
