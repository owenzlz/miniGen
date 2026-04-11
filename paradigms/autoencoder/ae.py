"""
AE (Autoencoder) generative process.

A deterministic autoencoder with pure reconstruction loss (no KL divergence).
This contrasts with VAE and illustrates why latent space regularization
matters for generation — sampling z ~ N(0, I) produces poor results because
the latent space is unregularized.

Training: Minimize MSE(x, decode(encode(x)))
Sampling: Reconstruct input data via encode -> decode.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE:
    """Autoencoder generative process.

    Wraps an AEModel to provide the standard process interface
    (sample_timesteps, training_loss, sample) used by train.py.

    Args:
        latent_dim: Latent space dimensionality (must match model).
    """

    def __init__(self, latent_dim: int = 16):
        self.latent_dim = latent_dim
        self.dataloader = None

    def set_dataloader(self, dataloader):
        """Store a reference to the dataloader for reconstruction sampling."""
        self.dataloader = dataloader

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (AE doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute reconstruction loss (MSE).

        Args:
            model: AEModel instance.
            x0: Clean data samples (B, D).
            t: Ignored (AE doesn't use timesteps).
        """
        x_recon = model(x0)
        return F.mse_loss(x_recon, x0)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Reconstruct data points from the dataloader.

        Encodes real data points and decodes them to test reconstruction.

        Args:
            model: AEModel instance.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Reconstructed samples (num_samples, data_dim).
        """
        model.eval()
        num_samples = shape[0]

        if self.dataloader is not None:
            # Collect real data from the dataloader
            sample_list = []
            for batch in self.dataloader:
                sample_list.append(batch["tenPoints"])
                if sum(s.shape[0] for s in sample_list) >= num_samples:
                    break
            x = torch.cat(sample_list, dim=0)[:num_samples].to(device)
            return model(x)

        # Fallback: decode from Gaussian noise
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return model.decode(z)
