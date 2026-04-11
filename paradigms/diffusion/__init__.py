"""
Diffusion module for denoising diffusion probabilistic models.

This package provides modular, extensible components for diffusion models:

- **Base classes**: `Schedule`, `DiffusionBase`, `Sampler`, `Parameterization`
- **Schedules**: `LinearBetaSchedule`, `CosineBetaSchedule`
- **Samplers**: `DDPMSampler`, `DDIMSampler`
- **Main class**: `DDPM` (ties components together with backward-compatible API)
- **Utilities**: `extract()`, `predict_parameterization` module

Example usage:
    from paradigms.diffusion import DDPM, LinearBetaSchedule, Parameterization

    schedule = LinearBetaSchedule(num_timesteps=1000).to(device)
    diffusion = DDPM(schedule, Parameterization.EPS)
    samples = diffusion.sample(model, shape, device)
"""

# Base classes and enums
from .base import Parameterization, Schedule, DiffusionBase, Sampler

# Noise schedules
from .scheduler import LinearBetaSchedule, CosineBetaSchedule

# Main DDPM class
from .ddpm import DDPM

# Samplers
from .sampler import DDPMSampler, DDIMSampler

# Utilities
from .utils import extract

# Pure functions for advanced users
from . import predict_parameterization

__all__ = [
    # Base
    "Parameterization",
    "Schedule",
    "DiffusionBase",
    "Sampler",
    # Schedules
    "LinearBetaSchedule",
    "CosineBetaSchedule",
    # DDPM
    "DDPM",
    # Samplers
    "DDPMSampler",
    "DDIMSampler",
    # Utils
    "extract",
    "predict_parameterization",
]
