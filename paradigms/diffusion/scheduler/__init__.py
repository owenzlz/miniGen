"""
Noise schedules for diffusion models.

Available schedules:
- LinearBetaSchedule: Linear β_t from DDPM (Ho et al., 2020)
- CosineBetaSchedule: Cosine ᾱ_t from Improved DDPM (Nichol & Dhariwal, 2021)

Future schedules can be added as separate modules (e.g., sigmoid.py, continuous.py).
"""

from .base import ScheduleBuffers
from .linear import LinearBetaSchedule
from .cosine import CosineBetaSchedule

__all__ = [
    "ScheduleBuffers",
    "LinearBetaSchedule",
    "CosineBetaSchedule",
]
