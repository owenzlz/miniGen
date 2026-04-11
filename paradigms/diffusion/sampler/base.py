"""
Shared utilities for samplers.

Notation (following Ho et al., 2020):
- α_t = 1 - β_t
- ᾱ_t = ∏_{s=1}^t α_s  (alphas_cumprod)
- x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε,  where ε ~ N(0, I)
"""
from typing import Tuple
import torch

from ..utils import extract


def q_posterior_mean_variance(
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    posterior_mean_coef1: torch.Tensor,
    posterior_mean_coef2: torch.Tensor,
    posterior_variance: torch.Tensor,
    posterior_log_variance_clipped: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute posterior q(x_{t-1} | x_t, x_0).

    From Eq. 6-7 in Ho et al. 2020:

        q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t(x_t, x_0), β̃_t I)

    where:
        μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

        β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t

    Args:
        x0: Predicted clean sample.
        xt: Noisy sample at timestep t.
        t: Timestep indices.
        posterior_mean_coef1: (√ᾱ_{t-1} β_t)/(1-ᾱ_t), coefficient for x_0.
        posterior_mean_coef2: (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t), coefficient for x_t.
        posterior_variance: β̃_t values.
        posterior_log_variance_clipped: log(β̃_t) clipped for numerical stability.

    Returns:
        Tuple of (μ̃_t, β̃_t, log β̃_t).
    """
    # μ̃_t = coef1 * x_0 + coef2 * x_t
    posterior_mean = (
        extract(posterior_mean_coef1, t, xt.shape) * x0
        + extract(posterior_mean_coef2, t, xt.shape) * xt
    )
    var = extract(posterior_variance, t, xt.shape)
    log_var = extract(posterior_log_variance_clipped, t, xt.shape)
    return posterior_mean, var, log_var
