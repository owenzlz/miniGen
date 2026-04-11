"""
GAN (Generative Adversarial Network) generative process.

Reference: Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.

Uses non-saturating GAN loss:
  D loss: -E[log D(x_real)] - E[log(1 - D(G(z)))]
  G loss: -E[log D(G(z))]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN:
    """GAN generative process.

    Manages the discriminator and its optimizer internally.
    The generator (model) and its optimizer are managed by train.py.

    Uses `train_step` instead of `training_loss` because GAN requires
    alternating generator/discriminator updates.

    Args:
        discriminator: Discriminator network.
        disc_optimizer: Optimizer for the discriminator.
        latent_dim: Latent noise dimensionality.
        n_critic: Number of discriminator updates per generator update.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        disc_optimizer: torch.optim.Optimizer,
        latent_dim: int = 16,
        n_critic: int = 1,
    ):
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.latent_dim = latent_dim
        self.n_critic = n_critic

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (GAN doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def train_step(
        self,
        generator: nn.Module,
        x_real: torch.Tensor,
        gen_optimizer: torch.optim.Optimizer,
        y=None,
        grad_clip: float = 1.0,
    ) -> float:
        """Perform one full GAN training step (D + G updates).

        Args:
            generator: Generator network.
            x_real: Real data samples (B, D).
            gen_optimizer: Optimizer for the generator.
            y: Class labels (ignored for unconditional GAN).
            grad_clip: Max gradient norm for generator.

        Returns:
            Generator loss value (float) for logging.
        """
        device = x_real.device
        batch_size = x_real.shape[0]

        # === Train Discriminator ===
        for _ in range(self.n_critic):
            z = torch.randn(batch_size, self.latent_dim, device=device)
            with torch.no_grad():
                x_fake = generator(z)

            d_real = self.discriminator(x_real)
            d_fake = self.discriminator(x_fake)

            d_loss_real = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            )
            d_loss_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            d_loss = d_loss_real + d_loss_fake

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

        # === Train Generator ===
        z = torch.randn(batch_size, self.latent_dim, device=device)
        x_fake = generator(z)
        d_fake = self.discriminator(x_fake)

        # Non-saturating loss: -E[log D(G(z))]
        g_loss = F.binary_cross_entropy_with_logits(
            d_fake, torch.ones_like(d_fake)
        )

        gen_optimizer.zero_grad()
        g_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        gen_optimizer.step()

        return g_loss.item()

    def training_loss(self, model, x0, t, **kwargs):
        """Not used for GAN. Use train_step instead."""
        raise NotImplementedError("GAN uses train_step, not training_loss")

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples from the generator.

        Args:
            model: Generator network.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Generated samples (num_samples, data_dim).
        """
        model.eval()
        num_samples = shape[0]
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return model(z)
