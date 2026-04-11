"""
DDPM (Denoising Diffusion Probabilistic Models) implementation.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

This module provides the main DDPM class which ties together schedules,
parameterizations, and samplers into a cohesive interface.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Schedule, DiffusionBase, Parameterization
from .utils import extract
from . import predict_parameterization as pred
from .sampler import DDPMSampler, q_posterior_mean_variance


class DDPM(DiffusionBase):
    """DDPM diffusion process.

    This class provides the core diffusion operations (forward process, training loss,
    reverse sampling) and convenience methods wrapping the pure functions in
    predict_parameterization.
    """

    def __init__(
        self,
        schedule: Schedule,
        parameterization: Parameterization = Parameterization.EPS,
    ):
        super().__init__(schedule, parameterization)
        # Internal sampler for default sample() method
        self._sampler = DDPMSampler(schedule, parameterization)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, T-1].

        Convenience method that delegates to the schedule.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with integer timesteps.
        """
        return self.schedule.sample_timesteps(batch_size, device)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod = extract(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute training loss based on parameterization."""
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        model_out = model(xt, t, **kwargs)

        target = pred.get_training_target(
            x0, noise, t, self.parameterization,
            self.schedule.sqrt_alphas_cumprod,
            self.schedule.sqrt_one_minus_alphas_cumprod,
        )

        loss = F.mse_loss(model_out, target)
        return loss

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        return q_posterior_mean_variance(
            x0, xt, t,
            self.schedule.posterior_mean_coef1,
            self.schedule.posterior_mean_coef2,
            self.schedule.posterior_variance,
            self.schedule.posterior_log_variance_clipped,
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute p(x_{t-1} | x_t) using learned model."""
        model_out = model(xt, t, **kwargs)

        # Get x0 prediction based on parameterization
        x0_pred = pred.model_output_to_x0(
            model_out, xt, t, self.parameterization,
            self.schedule.sqrt_alphas_cumprod,
            self.schedule.sqrt_one_minus_alphas_cumprod,
        )

        # Clip x0 to [-1, 1] for stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return self.q_posterior_mean_variance(x0_pred, xt, t)

    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        return self._sampler.p_sample(model, xt, t, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples using DDPM sampling."""
        return self._sampler.sample(model, shape, device, **kwargs)
