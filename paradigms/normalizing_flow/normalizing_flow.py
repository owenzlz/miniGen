"""
Normalizing Flow generative process.

Reference: Dinh et al., "Density estimation using Real-NVP", ICLR 2017.

Training: Maximize log-likelihood via change of variables.
    log p(x) = log p_z(f(x)) + log |det J_f(x)|
Sampling: z ~ N(0, I), x = f^{-1}(z)
"""
import math

import torch
import torch.nn as nn


class NormalizingFlow:
    """Normalizing Flow generative process.

    Wraps an invertible flow model (e.g., RealNVP2D) to provide the standard
    process interface. The model's forward pass maps data to latent space
    (returning z and log_det), and the inverse maps latent to data.
    """

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (NF doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Negative log-likelihood loss via change of variables.

        NLL = -E[log p_z(f(x)) + log |det J_f(x)|]
            = -E[log N(z; 0, I) + log_det]

        Args:
            model: Invertible flow model with forward(x) -> (z, log_det).
            x0: Data samples (B, D).
            t: Ignored.
        """
        z, log_det = model(x0)

        # Log-likelihood under standard normal prior
        log_prior = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)

        # log p(x) = log p(z) + log |det J|
        log_likelihood = log_prior + log_det

        return -log_likelihood.mean()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples by inverting the flow.

        z ~ N(0, I), x = f^{-1}(z)

        Args:
            model: Invertible flow model with inverse(z) -> x.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Generated samples.
        """
        model.eval()
        num_samples = shape[0]
        input_dim = shape[1] if len(shape) > 1 else model.input_dim
        z = torch.randn(num_samples, input_dim, device=device)
        return model.inverse(z)
