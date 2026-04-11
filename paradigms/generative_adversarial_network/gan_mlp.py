"""
GAN (Generative Adversarial Network) Generator and Discriminator MLPs for 2D data.

Reference: Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """MLP generator for 2D synthetic data.

    Maps latent noise z ~ N(0, I) to data space with tanh output.

    Args:
        latent_dim: Latent noise dimensionality.
        hidden_size: Hidden layer dimension.
        depth: Number of hidden layers.
        output_dim: Output data dimensionality (2 for 2D points).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_size: int = 256,
        depth: int = 4,
        output_dim: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent noise.

        Args:
            z: Latent noise (B, latent_dim).

        Returns:
            Generated samples (B, output_dim) in [-1, 1].
        """
        return self.net(z)


class Discriminator(nn.Module):
    """MLP discriminator for 2D synthetic data.

    Maps data points to real/fake logits.

    Args:
        input_dim: Input data dimensionality (2 for 2D points).
        hidden_size: Hidden layer dimension.
        depth: Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 4,
        **kwargs,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_size), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2)])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute real/fake logits.

        Args:
            x: Input data (B, input_dim).

        Returns:
            Logits (B, 1). Apply sigmoid for probabilities.
        """
        return self.net(x)
