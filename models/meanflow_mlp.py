"""
Single-head MLP for MeanFlow.

Architecture: conditions on both t (noise level) and h = t - r (interval width),
outputs a single average velocity u.

Reference: Geng et al., "Mean Flows for One-step Generative Modeling", arXiv:2505.13447.
"""
import torch
import torch.nn as nn

from .mlp import SinusoidalEmbedding, ResidualMLPBlock


class MeanFlowMLPDenoiser(nn.Module):
    """
    Single-head MLP denoiser for MeanFlow.

    Predicts the average velocity u(z_t, t, h) where h = t - r.

    Args:
        input_dim: Input dimension (2 for 2D points).
        hidden_size: Hidden layer dimension.
        depth: Number of residual blocks.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Time embeddings for t and h
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.h_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, hidden_size)
            for _ in range(depth)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.output_proj = nn.Linear(hidden_size, input_dim)

        self._init_weights()

    def _init_weights(self):
        """Zero-init output projection and AdaLN modulation layers."""
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.output_adaLN[-1].weight)
        nn.init.zeros_(self.output_adaLN[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, input_dim) noisy input points.
            t: (B,) noise level (time).
            h: (B,) time difference h = t - r.

        Returns:
            (B, input_dim) predicted average velocity u.
        """
        feat = self.input_proj(x)
        c = self.t_embed(t) + self.h_embed(h)

        for block in self.blocks:
            feat = block(feat, c)

        shift, scale = self.output_adaLN(c).chunk(2, dim=-1)
        feat = self.output_norm(feat) * (1 + scale) + shift
        return self.output_proj(feat)
