"""
Sampling strategies for diffusion models.

Available samplers:
- DDPMSampler: Standard DDPM sampling using all timesteps
- DDIMSampler: Accelerated/deterministic sampling with configurable steps and eta

Future samplers can be added as separate modules (e.g., dpm_solver.py, ode.py).
"""

from .base import q_posterior_mean_variance
from .ddpm_sampler import DDPMSampler
from .ddim_sampler import DDIMSampler

__all__ = [
    "q_posterior_mean_variance",
    "DDPMSampler",
    "DDIMSampler",
]
