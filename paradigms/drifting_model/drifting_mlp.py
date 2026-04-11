"""
Drifting Model Generator MLP for 2D data.

Reference: Deng et al., "Generative Modeling via Drifting", 2026.

Simple MLP that maps latent noise z ~ N(0, I) to data space.
Uses SiLU activations and no output activation (unlike GAN's Tanh).
"""
import torch
import torch.nn as nn


class DriftingGenerator(nn.Module):
    """MLP generator for drifting models on 2D synthetic data.

    Args:
        latent_dim: Latent noise dimensionality.
        hidden_size: Hidden layer dimension.
        depth: Number of hidden layers.
        output_dim: Output data dimensionality (2 for 2D points).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 256,
        depth: int = 3,
        output_dim: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim, hidden_size), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent noise.

        Args:
            z: Latent noise (B, latent_dim).

        Returns:
            Generated samples (B, output_dim).
        """
        return self.net(z)
