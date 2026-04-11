"""
MLP-based denoiser for 2D diffusion experiments.

Simple architecture with time conditioning via adaptive layer norm,
similar to DiT but without the transformer/patch machinery.
"""
import math
from typing import Optional
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timesteps

        Returns:
            (B, dim) embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with adaptive layer norm conditioning.

    Uses AdaLN-Zero style modulation: shift, scale, and gate.
    """

    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        # AdaLN modulation: shift, scale, gate
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, hidden_size * 3),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_size) input
            c: (B, cond_size) conditioning

        Returns:
            (B, hidden_size) output
        """
        shift, scale, gate = self.adaLN(c).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.mlp(h)
        return x + gate * h


class MLPDenoiser(nn.Module):
    """
    MLP-based denoiser for 2D diffusion.

    Architecture:
    - Input projection: 2D -> hidden_size
    - Timestep + class embeddings
    - Stack of residual MLP blocks with AdaLN conditioning
    - Output projection: hidden_size -> 2D

    Args:
        input_dim: Input dimension (2 for 2D points).
        hidden_size: Hidden layer dimension.
        depth: Number of residual blocks.
        num_classes: Number of classes for conditioning (0 for unconditional).
        class_dropout_prob: Probability of dropping class labels for CFG.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        num_classes: int = 0,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Class embedding (if conditional)
        if num_classes > 0:
            # +1 for null class (CFG)
            self.class_embed = nn.Embedding(num_classes + 1, hidden_size)
        else:
            self.class_embed = None

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, hidden_size)
            for _ in range(depth)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, input_dim)

        # Final AdaLN for output
        self.output_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Zero-init output projection for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Zero-init AdaLN modulation layers
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.output_adaLN[-1].weight)
        nn.init.zeros_(self.output_adaLN[-1].bias)

    def _drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Randomly drop labels for classifier-free guidance training."""
        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
            labels = torch.where(drop_mask, self.num_classes, labels)
        return labels

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        uncond: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, input_dim) noisy input points
            t: (B,) timesteps
            y: (B,) class labels (optional)
            uncond: If True, zero out time conditioning (for equilibrium matching).

        Returns:
            (B, input_dim) predicted noise / x0 / v
        """
        # Input projection
        h = self.input_proj(x)

        # Conditioning: time + class
        if uncond:
            c = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            c = self.time_embed(t)
        if self.class_embed is not None and y is not None:
            y = self._drop_labels(y)
            c = c + self.class_embed(y)

        # Residual blocks
        for block in self.blocks:
            h = block(h, c)

        # Output
        shift, scale = self.output_adaLN(c).chunk(2, dim=-1)
        h = self.output_norm(h) * (1 + scale) + shift
        return self.output_proj(h)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.

        Args:
            x: (B, input_dim) noisy input points
            t: (B,) timesteps
            y: (B,) class labels
            cfg_scale: Guidance scale (1.0 = no guidance)

        Returns:
            (B, input_dim) guided prediction
        """
        # Conditional prediction
        cond_out = self.forward(x, t, y)

        if cfg_scale == 1.0:
            return cond_out

        # Unconditional prediction (null class)
        y_null = torch.full_like(y, self.num_classes)
        uncond_out = self.forward(x, t, y_null)

        # CFG combination
        return uncond_out + cfg_scale * (cond_out - uncond_out)
