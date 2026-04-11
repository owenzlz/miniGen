"""
Linear beta schedule from DDPM paper.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""
import torch

from ..base import Schedule
from .base import ScheduleBuffers


class LinearBetaSchedule(Schedule, ScheduleBuffers):
    """Linear beta schedule from DDPM paper (Ho et al., 2020).

    β_t increases linearly from β_start to β_end:
        β_t = β_start + (β_end - β_start) * (t-1) / (T-1)

    Default values β_start=1e-4, β_end=0.02 are from the original paper.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """Initialize linear beta schedule.

        Args:
            num_timesteps: T, total number of diffusion steps.
            beta_start: β_1, variance at first timestep (small, ~1e-4).
            beta_end: β_T, variance at final timestep (larger, ~0.02).
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # β_t ∈ [β_start, β_end], linearly spaced
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self._compute_buffers(betas)

    def get_alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get ᾱ_t at timestep t."""
        return self.alphas_cumprod[t]

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Get signal-to-noise ratio: SNR(t) = ᾱ_t / (1-ᾱ_t)."""
        alpha_cumprod = self.alphas_cumprod[t]
        return alpha_cumprod / (1.0 - alpha_cumprod)

    def to(self, device):
        """Move all tensors to device."""
        for name in self._buffer_names():
            setattr(self, name, getattr(self, name).to(device))
        return self
