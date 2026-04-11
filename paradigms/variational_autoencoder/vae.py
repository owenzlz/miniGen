"""
VAE (Variational Autoencoder) generative process.

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.

Training: Maximize ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
Sampling: z ~ N(0, I), x = decode(z)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE:
    """VAE generative process.

    Wraps a VAEModel to provide the standard process interface
    (sample_timesteps, training_loss, sample) used by train.py.

    Args:
        latent_dim: Latent space dimensionality (must match model).
        beta: Weight for KL divergence term (beta-VAE). 1.0 = standard ELBO.
    """

    def __init__(self, latent_dim: int = 16, beta: float = 1.0):
        self.latent_dim = latent_dim
        self.beta = beta

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (VAE doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute negative ELBO loss.

        Loss = reconstruction_loss + beta * KL_divergence

        Args:
            model: VAEModel instance.
            x0: Clean data samples (B, D).
            t: Ignored (VAE doesn't use timesteps).
        """
        x_recon, mu, log_var = model(x0)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x0)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + self.beta * kl_loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples by decoding random latent codes.

        Args:
            model: VAEModel instance.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Generated samples (num_samples, data_dim).
        """
        model.eval()
        num_samples = shape[0]
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return model.decode(z)
