"""
Base utilities for noise schedules.

================================================================================
DDPM NOTATION (Ho et al., 2020)
================================================================================

Forward process (Eq. 2):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

Closed-form forward (Eq. 4):
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

    where:
        α_t = 1 - β_t
        ᾱ_t = ∏_{s=1}^t α_s

    Reparameterization:
        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε,  where ε ~ N(0, I)

Reverse posterior (Eq. 6-7):
    q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t(x_t, x_0), β̃_t I)

    where:
        μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t

        β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t

Signal-to-noise ratio:
    SNR(t) = ᾱ_t / (1-ᾱ_t)
"""
import torch
import torch.nn.functional as F


class ScheduleBuffers:
    """Mixin for computing and managing schedule buffers.

    This eliminates duplication between schedule implementations by providing
    a single method to compute all derived quantities from betas.
    """

    def _compute_buffers(self, betas: torch.Tensor) -> None:
        """Compute all schedule buffers from betas.

        Args:
            betas: β_t values for each timestep, shape (T,).
                   Controls variance of forward process q(x_t | x_{t-1}).

                   β_t is the small noise increments (e.g., 1e-4 to 0.02).
                   x_t = √(1-β_t) x_{t-1} + √(β_t) ε, ε ~ N(0, I)
                   or equivalently:
                   x_t = √α_t x_{t-1} + √(1-α_t) ε, where α_t = 1 - β_t.
        """
        # ──────────────────────────────────────────────────────────────────────
        # Core schedule values
        # ──────────────────────────────────────────────────────────────────────

        # α_t = 1 - β_t
        # Scaling factor for mean in q(x_t | x_{t-1}) = N(√α_t x_{t-1}, β_t I)
        alphas = 1.0 - betas

        # ᾱ_t = ∏_{s=1}^t α_s = α_1 * α_2 * ... * α_t
        # Cumulative product of alphas, used in closed-form q(x_t | x_0)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # ᾱ_{t-1} (with ᾱ_0 = 1 by convention)
        # Needed for posterior mean coefficients
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Helper to register buffers as attributes
        register = lambda name, val: setattr(self, name, val)

        # Store core values
        register("betas", betas)                      # β_t
        register("alphas", alphas)                    # α_t = 1 - β_t
        register("alphas_cumprod", alphas_cumprod)    # ᾱ_t = ∏_{s=1}^t α_s
        register("alphas_cumprod_prev", alphas_cumprod_prev)  # ᾱ_{t-1}

        # ──────────────────────────────────────────────────────────────────────
        # Forward process q(x_t | x_0) coefficients
        # ──────────────────────────────────────────────────────────────────────
        # From Eq. 4: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

        # √ᾱ_t : coefficient for x_0 in forward diffusion
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))

        # √(1-ᾱ_t) : coefficient for noise ε in forward diffusion
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # ──────────────────────────────────────────────────────────────────────
        # Posterior q(x_{t-1} | x_t, x_0) coefficients (Eq. 6-7)
        # ──────────────────────────────────────────────────────────────────────
        # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, β̃_t I)

        # β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        # Posterior variance (Eq. 7)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register("posterior_variance", posterior_variance)

        # log(β̃_t), clipped for numerical stability (avoid log(0) at t=0)
        register("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        # μ̃_t = coef1 * x_0 + coef2 * x_t  (Eq. 7)
        #
        # coef1 = (√ᾱ_{t-1} * β_t) / (1-ᾱ_t)
        # coef2 = (√α_t * (1-ᾱ_{t-1})) / (1-ᾱ_t)

        # Coefficient for x_0 in posterior mean
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register("posterior_mean_coef1", posterior_mean_coef1)

        # Coefficient for x_t in posterior mean
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        register("posterior_mean_coef2", posterior_mean_coef2)

    def _buffer_names(self) -> list:
        """Return list of buffer attribute names for device transfer."""
        return [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance", "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2"
        ]
