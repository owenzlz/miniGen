"""
Cosine beta schedule from Improved DDPM paper.

Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.
"""
import math
import torch

from ..base import Schedule
from .base import ScheduleBuffers


class CosineBetaSchedule(Schedule, ScheduleBuffers):
    """Cosine beta schedule from Improved DDPM (Nichol & Dhariwal, 2021).

    Instead of defining β_t directly, this schedule defines ᾱ_t via a cosine:

        f(t) = cos²((t/T + s) / (1 + s) * π/2)
        ᾱ_t = f(t) / f(0)

    Then β_t is derived: β_t = 1 - ᾱ_t / ᾱ_{t-1}

    The cosine schedule provides:
    - Gentler noise addition near t=0 (preserves signal longer)
    - Gentler noise addition near t=T (smoother transition to pure noise)
    - Better sample quality than linear schedule

    The offset s (default 0.008) prevents β_t from being too small near t=0.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
    ):
        """Initialize cosine beta schedule.

        Args:
            num_timesteps: T, total number of diffusion steps.
            s: Small offset to prevent β_t ≈ 0 near t=0.
        """
        self.num_timesteps = num_timesteps
        self.s = s

        # ──────────────────────────────────────────────────────────────────────
        # Compute ᾱ_t from cosine schedule (Eq. 17 in Nichol & Dhariwal)
        # ──────────────────────────────────────────────────────────────────────
        # f(t) = cos²((t/T + s) / (1 + s) * π/2)
        # ᾱ_t = f(t) / f(0)

        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps) / num_timesteps  # t ∈ [0, 1]

        # f(t) = cos²(...)
        f_t = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2

        # Normalize: ᾱ_t = f(t) / f(0) so that ᾱ_0 = 1
        alphas_cumprod = f_t / f_t[0]

        # ──────────────────────────────────────────────────────────────────────
        # Derive β_t from ᾱ_t
        # ──────────────────────────────────────────────────────────────────────
        # From ᾱ_t = ∏_{s=1}^t α_s and α_t = 1 - β_t:
        #   α_t = ᾱ_t / ᾱ_{t-1}
        #   β_t = 1 - α_t = 1 - ᾱ_t / ᾱ_{t-1}

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clamp to prevent numerical issues (β_t should be in (0, 1))
        betas = torch.clamp(betas, 0.0, 0.999)

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
