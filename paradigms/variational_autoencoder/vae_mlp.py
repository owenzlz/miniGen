"""
VAE (Variational Autoencoder) with MLP encoder/decoder for 2D data.

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
"""
import torch
import torch.nn as nn


class VAEModel(nn.Module):
    """MLP-based VAE for 2D synthetic data.

    Architecture:
    - Encoder: MLP mapping input_dim -> (mu, log_var) of latent_dim
    - Decoder: MLP mapping latent_dim -> input_dim (tanh output for [-1, 1])
    - Reparameterization trick for differentiable sampling

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
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_dim)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            dec_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        dec_layers.append(nn.Linear(hidden_size, input_dim))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        """Encode input to latent distribution parameters.

        Returns:
            Tuple of (mu, log_var), each shape (B, latent_dim).
        """
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to data space. Output in [-1, 1]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs):
        """Full forward pass: encode, reparameterize, decode.

        Returns:
            Tuple of (x_recon, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
