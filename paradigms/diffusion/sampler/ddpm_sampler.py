"""
DDPM Sampler - Standard sampling using all timesteps.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn

from ..base import Sampler, Schedule, Parameterization
from .. import predict_parameterization as pred
from .base import q_posterior_mean_variance


class DDPMSampler(Sampler):
    """Standard DDPM sampler using all timesteps.

    Implements the original DDPM sampling algorithm (Algorithm 2 in Ho et al. 2020)
    which iterates through all timesteps from T to 0.

    The reverse process is:
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    where μ_θ is computed from the model's prediction of x_0 or ε.
    """

    def __init__(
        self,
        schedule: Schedule,
        parameterization: Parameterization = Parameterization.EPS,
    ):
        super().__init__(schedule, parameterization)

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0). See base.q_posterior_mean_variance."""
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
        """Compute p_θ(x_{t-1} | x_t) using learned model.

        First predicts x_0 from model output, then computes the posterior.
        For ε-prediction (Eq. 11 in Ho et al. 2020):

            x̂_0 = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t

        Then uses q(x_{t-1} | x_t, x̂_0) as p_θ(x_{t-1} | x_t).
        """
        model_out = model(xt, t, **kwargs)

        # Get x̂_0 prediction based on parameterization
        x0_pred = pred.model_output_to_x0(
            model_out, xt, t, self.parameterization,
            self.schedule.sqrt_alphas_cumprod,
            self.schedule.sqrt_one_minus_alphas_cumprod,
        )

        # Clip x̂_0 to [-1, 1] for stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return self.q_posterior_mean_variance(x0_pred, xt, t)

    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

        From Algorithm 2 in Ho et al. 2020:

            x_{t-1} = μ_θ(x_t, t) + σ_t * z,  where z ~ N(0, I)

        σ_t = √β̃_t (we use the posterior variance).
        No noise is added when t = 0.
        """
        batch_size = xt.shape[0]

        # μ_θ and log(σ_t²) = log(β̃_t)
        mean, _, log_variance = self.p_mean_variance(model, xt, t, **kwargs)

        # z ~ N(0, I)
        noise = torch.randn_like(xt)

        # No noise when t = 0
        nonzero_mask = (t != 0).float().view(batch_size, *((1,) * (len(xt.shape) - 1)))

        # x_{t-1} = μ_θ + σ_t z = μ_θ + exp(0.5 log β̃_t) z
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples using DDPM sampling (Algorithm 2).

        Starting from x_T ~ N(0, I), iteratively sample:
            x_{t-1} ~ p_θ(x_{t-1} | x_t)  for t = T, T-1, ..., 1

        Args:
            model: The denoising model ε_θ or x_θ.
            shape: Shape of samples to generate.
            device: Device to generate on.
            num_steps: Ignored for DDPM (uses all T timesteps).
            **kwargs: Additional arguments passed to model.

        Returns:
            Generated samples x_0.
        """
        _ = num_steps  # DDPM uses all timesteps
        model.eval()
        batch_size = shape[0]

        # x_T ~ N(0, I)
        x = torch.randn(shape, device=device)

        # Iterate from t = T-1 down to t = 0
        for t in reversed(range(self.schedule.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, **kwargs)

        return x
