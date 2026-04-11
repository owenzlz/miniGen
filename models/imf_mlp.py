"""
Dual-head MLP for Improved MeanFlow (iMF).

Architecture: shared backbone + separate u-head (average velocity) and v-head
(instantaneous velocity). Conditions on both t (noise level) and h = t - r
(interval width).

Note: The reference iMF paper conditions only on h, because for high-dimensional
data (images) the noise level can be inferred from z_t. For low-dimensional (2D)
data, explicit t conditioning is necessary.

Reference: Geng et al., "Improved Mean Flows: On the Challenges of Fastforward Generative Models", arXiv:2512.02012.
"""
import torch
import torch.nn as nn

from .mlp import SinusoidalEmbedding, ResidualMLPBlock


class iMFMLPDenoiser(nn.Module):
    """
    Dual-head MLP denoiser for Improved MeanFlow.

    Shared backbone produces a representation conditioned on t and h = t - r,
    then two separate heads predict:
      - u: average velocity field
      - v: instantaneous velocity field

    Args:
        input_dim: Input dimension (2 for 2D points).
        hidden_size: Hidden layer dimension.
        shared_depth: Number of shared residual blocks.
        head_depth: Number of residual blocks per head.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        shared_depth: int = 2,
        head_depth: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Time embedding for h = t - r
        self.h_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Time embedding for t (noise level)
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Shared backbone
        self.shared_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, hidden_size)
            for _ in range(shared_depth)
        ])

        # u-head (average velocity)
        self.u_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, hidden_size)
            for _ in range(head_depth)
        ])
        self.u_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.u_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.u_proj = nn.Linear(hidden_size, input_dim)

        # v-head (instantaneous velocity)
        self.v_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, hidden_size)
            for _ in range(head_depth)
        ])
        self.v_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.v_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.v_proj = nn.Linear(hidden_size, input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Zero-init output projections
        nn.init.zeros_(self.u_proj.weight)
        nn.init.zeros_(self.u_proj.bias)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

        # Zero-init AdaLN modulation layers
        for block in self.shared_blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        for block in self.u_blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        for block in self.v_blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.u_adaLN[-1].weight)
        nn.init.zeros_(self.u_adaLN[-1].bias)
        nn.init.zeros_(self.v_adaLN[-1].weight)
        nn.init.zeros_(self.v_adaLN[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, input_dim) noisy input points
            t: (B,) noise level (time)
            h: (B,) time difference h = t - r

        Returns:
            (u, v) where both are (B, input_dim)
        """
        # Input projection
        feat = self.input_proj(x)

        # Conditioning on both t and h
        c = self.h_embed(h) + self.t_embed(t)

        # Shared backbone
        for block in self.shared_blocks:
            feat = block(feat, c)

        # u-head
        u = feat
        for block in self.u_blocks:
            u = block(u, c)
        shift, scale = self.u_adaLN(c).chunk(2, dim=-1)
        u = self.u_norm(u) * (1 + scale) + shift
        u = self.u_proj(u)

        # v-head
        v = feat
        for block in self.v_blocks:
            v = block(v, c)
        shift, scale = self.v_adaLN(c).chunk(2, dim=-1)
        v = self.v_norm(v) * (1 + scale) + shift
        v = self.v_proj(v)

        return u, v
