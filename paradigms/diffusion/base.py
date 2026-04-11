"""
Base interfaces for diffusion/flow models.

Design principle: Separate parameterization, schedule, and sampler.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional
import torch
import torch.nn as nn


class Parameterization(Enum):
    """What the network predicts."""
    EPS = "eps"      # noise
    X0 = "x0"        # clean data
    V = "v"          # velocity (for v-prediction)
    FLOW = "flow"    # flow velocity (for flow matching)


class Schedule(ABC):
    """Abstract schedule interface."""

    @abstractmethod
    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative product of alpha at timestep t."""
        pass

    @abstractmethod
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Get signal-to-noise ratio at timestep t."""
        pass

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class DiffusionBase(ABC, nn.Module):
    """Base class for diffusion/flow processes."""

    def __init__(
        self,
        schedule: Schedule,
        parameterization: Parameterization = Parameterization.EPS,
    ):
        super().__init__()
        self.schedule = schedule
        self.parameterization = parameterization

    @abstractmethod
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to x0."""
        pass

    @abstractmethod
    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute training loss."""
        pass

    @abstractmethod
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Reverse one step."""
        pass

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples from noise."""
        pass


class Sampler(ABC):
    """Abstract sampler interface for decoupled sampling strategies.

    Samplers hold a reference to the schedule and parameterization, enabling
    them to perform sampling independently of a specific diffusion instance.
    """

    def __init__(
        self,
        schedule: Schedule,
        parameterization: Parameterization = Parameterization.EPS,
    ):
        """Initialize sampler with schedule and parameterization.

        Args:
            schedule: Noise schedule providing alpha/beta values.
            parameterization: What the model predicts (EPS, X0, V).
        """
        self.schedule = schedule
        self.parameterization = parameterization

    @abstractmethod
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Perform a single denoising step.

        Args:
            model: The denoising model.
            xt: Noisy sample at timestep t.
            t: Current timestep indices.
            **kwargs: Additional arguments passed to model.

        Returns:
            Denoised sample at timestep t-1.
        """
        pass

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples from noise.

        Args:
            model: The denoising model.
            shape: Shape of samples to generate (batch_size, ...).
            device: Device to generate on.
            num_steps: Number of sampling steps (None = use all timesteps).
            **kwargs: Additional arguments passed to model.

        Returns:
            Generated samples.
        """
        pass
