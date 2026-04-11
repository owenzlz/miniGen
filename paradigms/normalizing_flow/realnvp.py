"""
RealNVP (Real-valued Non-Volume Preserving) normalizing flow for 2D data.

Reference: Dinh et al., "Density estimation using Real-NVP", ICLR 2017.

For 2D data, each affine coupling layer transforms one dimension
conditioned on the other, alternating which dimension is fixed.
"""
import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for 2D data.

    Splits the 2D input into a fixed dimension and a transformed dimension.
    The transformation is: y_change = x_change * exp(s) + t
    where s, t = net(x_fixed).

    Args:
        mask_idx: Index of the dimension to keep fixed (0 or 1).
        hidden_size: Hidden layer dimension in the scale/translation network.
        depth: Number of hidden layers.
    """

    def __init__(self, mask_idx: int, hidden_size: int = 128, depth: int = 2):
        super().__init__()
        self.mask_idx = mask_idx
        self.change_idx = 1 - mask_idx

        # Network: 1D -> 2D (scale, translation)
        layers = [nn.Linear(1, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward pass: data -> latent.

        Returns:
            Tuple of (y, log_det) where log_det is per-sample scalar.
        """
        x_fixed = x[:, self.mask_idx : self.mask_idx + 1]  # (B, 1)
        x_change = x[:, self.change_idx]  # (B,)

        st = self.net(x_fixed)  # (B, 2)
        s = torch.tanh(st[:, 0])  # Bounded scale for stability
        t = st[:, 1]

        y_change = x_change * torch.exp(s) + t

        y = x.clone()
        y[:, self.change_idx] = y_change

        return y, s  # log_det = s

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass: latent -> data."""
        y_fixed = y[:, self.mask_idx : self.mask_idx + 1]  # (B, 1)
        y_change = y[:, self.change_idx]  # (B,)

        st = self.net(y_fixed)  # (B, 2)
        s = torch.tanh(st[:, 0])
        t = st[:, 1]

        x_change = (y_change - t) * torch.exp(-s)

        x = y.clone()
        x[:, self.change_idx] = x_change

        return x


class RealNVP2D(nn.Module):
    """RealNVP normalizing flow for 2D data.

    Stacks affine coupling layers with alternating masks.
    Forward: data -> latent (for training, compute log-likelihood).
    Inverse: latent -> data (for sampling).

    Args:
        input_dim: Input dimensionality (must be 2).
        hidden_size: Hidden size for coupling layer networks.
        num_coupling_layers: Number of coupling layers.
        coupling_depth: Number of hidden layers per coupling network.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 128,
        num_coupling_layers: int = 8,
        coupling_depth: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    mask_idx=i % 2,
                    hidden_size=hidden_size,
                    depth=coupling_depth,
                )
                for i in range(num_coupling_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """Forward pass: data -> latent with log-determinant.

        Returns:
            Tuple of (z, log_det_total) where z is in latent space.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: latent -> data (for sampling)."""
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z
