"""
Energy network for Energy-Based Models.

Maps x (B, 2) -> scalar energy (B,). The data distribution is modeled as
p(x) ~ exp(-E(x)), so low energy = high probability.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _maybe_sn(layer: nn.Module, use_spectral_norm: bool) -> nn.Module:
    """Optionally wrap a layer with spectral normalization."""
    return spectral_norm(layer) if use_spectral_norm else layer


class EnergyMLP(nn.Module):
    """MLP that outputs a scalar energy for each input point.

    Architecture: input projection -> residual MLP blocks with SiLU -> scalar output.

    When use_spectral_norm=True, all linear layers are spectrally normalized to
    constrain the Lipschitz constant (Du & Mordatch 2019).

    Args:
        input_dim: Dimension of input data (default 2).
        hidden_size: Width of hidden layers.
        depth: Number of residual blocks.
        use_spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        sn = use_spectral_norm
        self.input_proj = _maybe_sn(nn.Linear(input_dim, hidden_size), sn)

        blocks = []
        for _ in range(depth):
            blocks.append(_maybe_sn(nn.Linear(hidden_size, hidden_size), sn))
            blocks.append(nn.SiLU())
            blocks.append(_maybe_sn(nn.Linear(hidden_size, hidden_size), sn))
        self.blocks = nn.ModuleList(blocks)
        self.depth = depth

        self.output_proj = _maybe_sn(nn.Linear(hidden_size, 1), sn)

    def forward(self, x: torch.Tensor, t=None, **kwargs) -> torch.Tensor:
        """Compute scalar energy for each input.

        Args:
            x: Input data (B, input_dim).
            t: Ignored (for interface compatibility).

        Returns:
            Energy values (B,).
        """
        h = self.input_proj(x)
        for i in range(self.depth):
            base = i * 3
            residual = h
            h = self.blocks[base](h)      # Linear
            h = self.blocks[base + 1](h)   # SiLU
            h = self.blocks[base + 2](h)   # Linear
            h = h + residual
        energy = self.output_proj(h).squeeze(-1)
        return energy
