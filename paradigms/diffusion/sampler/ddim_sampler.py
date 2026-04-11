"""
DDIM Sampler - Accelerated/deterministic sampling.

Reference: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn

from ..base import Sampler, Schedule, Parameterization
from ..utils import extract
from .. import predict_parameterization as pred


class DDIMSampler(Sampler):
    """DDIM sampler for accelerated/deterministic sampling.

    Implements DDIM (Denoising Diffusion Implicit Models) which allows:
    - Fewer sampling steps than training timesteps
    - Deterministic sampling when η=0
    - Interpolation between DDPM (η=1) and deterministic (η=0)

    From Eq. 12 in Song et al. 2020, the generative process is:

        x_{t-1} = √ᾱ_{t-1} f_θ(x_t, t)
                  + √(1 - ᾱ_{t-1} - σ_t²) ε_θ(x_t, t)
                  + σ_t ε_t

    where:
        f_θ(x_t, t) = (x_t - √(1-ᾱ_t) ε_θ(x_t, t)) / √ᾱ_t  (predicted x_0)

        σ_t = η √((1-ᾱ_{t-1})/(1-ᾱ_t)) √(1 - ᾱ_t/ᾱ_{t-1})

        ε_t ~ N(0, I)

    When η=0, the process is deterministic (σ_t=0).
    When η=1, it recovers DDPM sampling.
    """

    def __init__(
        self,
        schedule: Schedule,
        parameterization: Parameterization = Parameterization.EPS,
        eta: float = 0.0,
    ):
        """Initialize DDIM sampler.

        Args:
            schedule: Noise schedule.
            parameterization: What the model predicts.
            eta: η, stochasticity parameter. η=0 is deterministic, η=1 is DDPM-like.
        """
        super().__init__(schedule, parameterization)
        self.eta = eta

    def _get_timestep_schedule(self, num_steps: int) -> torch.Tensor:
        """Get evenly spaced timesteps for accelerated sampling.

        Returns a subsequence t_S > t_{S-1} > ... > t_1 of [0, T-1],
        where S = num_steps. Timesteps go from high to low.
        """
        step_ratio = self.schedule.num_timesteps // num_steps
        timesteps = torch.arange(0, num_steps) * step_ratio
        timesteps = torch.flip(timesteps, [0])  # Reverse: high to low
        return timesteps

    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Perform one DDIM step: x_t → x_{t_prev}.

        From Eq. 12 in Song et al. 2020:

            x_{t-1} = √ᾱ_{t-1} x̂_0 + √(1 - ᾱ_{t-1} - σ_t²) ε_θ + σ_t ε

        where:
            x̂_0 = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t  (predicted x_0)

            σ_t = η √((1-ᾱ_{t-1})/(1-ᾱ_t)) √(1 - ᾱ_t/ᾱ_{t-1})
                = η √((1-ᾱ_{t-1})(ᾱ_{t-1} - ᾱ_t) / ((1-ᾱ_t) ᾱ_{t-1}))

        Args:
            model: The denoising model.
            xt: Noisy sample x_t.
            t: Current timestep indices.
            t_prev: Target timestep indices (defaults to t-1).
            **kwargs: Additional arguments passed to model.

        Returns:
            Sample x_{t_prev}.
        """
        batch_size = xt.shape[0]

        # Default t_prev to t-1
        if t_prev is None:
            t_prev = t - 1
            t_prev = torch.clamp(t_prev, min=0)

        # Get model output ε_θ(x_t, t)
        model_out = model(xt, t, **kwargs)

        # Convert to x̂_0 = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
        x0_pred = pred.model_output_to_x0(
            model_out, xt, t, self.parameterization,
            self.schedule.sqrt_alphas_cumprod,
            self.schedule.sqrt_one_minus_alphas_cumprod,
        )
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Re-derive ε_θ from x̂_0 (ensures consistency)
        # ε_θ = (x_t - √ᾱ_t x̂_0) / √(1-ᾱ_t)
        eps_pred = pred.predict_eps_from_x0(
            xt, t, x0_pred,
            self.schedule.sqrt_alphas_cumprod,
            self.schedule.sqrt_one_minus_alphas_cumprod,
        )

        # Get ᾱ_t and ᾱ_{t-1}
        alpha_cumprod_t = extract(self.schedule.alphas_cumprod, t, xt.shape)
        alpha_cumprod_prev = extract(self.schedule.alphas_cumprod, t_prev, xt.shape)

        # σ_t = η √((1-ᾱ_{t-1})/(1-ᾱ_t)) √(1 - ᾱ_t/ᾱ_{t-1})
        sigma = self.eta * torch.sqrt(
            (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_prev)
        )

        # "Predicted direction pointing to x_t": √(1 - ᾱ_{t-1} - σ_t²) ε_θ
        pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * eps_pred

        # x_{t-1} = √ᾱ_{t-1} x̂_0 + pred_dir
        x_prev = torch.sqrt(alpha_cumprod_prev) * x0_pred + pred_dir

        # Add noise: + σ_t ε  (only if η > 0 and not at final step)
        if self.eta > 0:
            noise = torch.randn_like(xt)
            nonzero_mask = (t_prev != 0).float().view(batch_size, *((1,) * (len(xt.shape) - 1)))
            x_prev = x_prev + nonzero_mask * sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples using DDIM sampling.

        Uses a subsequence of S timesteps where S <= T, allowing faster
        sampling with fewer steps. Reverse process goes from high to low:

            x_{t_S} -> x_{t_{S-1}} -> ... -> x_{t_1} -> x_{t_0}

        Args:
            model: The denoising model.
            shape: Shape of samples to generate.
            device: Device to generate on.
            num_steps: Number of sampling steps S (defaults to T).
            **kwargs: Additional arguments passed to model.

        Returns:
            Generated samples x_0.
        """
        model.eval()
        batch_size = shape[0]

        # Default to all timesteps
        if num_steps is None:
            num_steps = self.schedule.num_timesteps

        # Get timestep subsequence: t_S > t_{S-1} > ... > t_1
        timesteps = self._get_timestep_schedule(num_steps).to(device)

        # x_{t_T} ~ N(0, I)
        x = torch.randn(shape, device=device)

        # Iterate: t_S -> t_{S-1} -> ... -> t_1 -> t_0
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get next timestep (or 0 if at final step)
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.tensor(0, device=device)
            t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

            x = self.p_sample(model, x, t_batch, t_prev_batch, **kwargs)

        return x
