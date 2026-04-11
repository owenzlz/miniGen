"""
Autoencoder with MLP encoder/decoder for 2D data.

Unlike a VAE, the encoder maps directly to a latent code (no mu/log_var split,
no reparameterization trick). Training uses pure reconstruction loss.
"""
import torch
import torch.nn as nn


class AEModel(nn.Module):
    """MLP-based Autoencoder for 2D synthetic data.

    Architecture:
    - Encoder: MLP mapping input_dim -> latent_dim (single linear head)
    - Decoder: MLP mapping latent_dim -> input_dim (tanh output for [-1, 1])

    Args:
        input_dim: Input data dimensionality (2 for 2D points).
        hidden_size: Hidden layer dimension.
        depth: Number of hidden layers in encoder/decoder.
        latent_dim: Latent space dimensionality.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 3,
        latent_dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            enc_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_latent = nn.Linear(hidden_size, latent_dim)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            dec_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        dec_layers.append(nn.Linear(hidden_size, input_dim))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent code.

        Returns:
            Latent code z of shape (B, latent_dim).
        """
        h = self.encoder_body(x)
        return self.fc_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to data space. Output in [-1, 1]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Returns:
            Reconstruction x_recon of shape (B, input_dim).
        """
        z = self.encode(x)
        return self.decode(z)
