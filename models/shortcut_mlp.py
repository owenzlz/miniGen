"""
ShortcutMLPDenoiser: MLPDenoiser with step-size conditioning for Shortcut Models.

Extends the base MLPDenoiser by adding a learnable step-size embedding that is
summed into the conditioning vector alongside the timestep embedding.

Reference:
- Frans et al., "One Step Diffusion via Shortcut Models", 2024.
"""
from typing import Optional

import torch
import torch.nn as nn

from .mlp import MLPDenoiser, SinusoidalEmbedding


class ShortcutMLPDenoiser(MLPDenoiser):
    """MLPDenoiser with additional step-size `d` conditioning.

    The conditioning vector becomes: c = time_embed(t) + step_embed(d) [+ class_embed(y)].
    When d is None (or not provided), it defaults to zero, recovering the
    standard MLPDenoiser behavior.

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
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
        )

        # Step-size embedding (same structure as time_embed)
        self.step_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Zero-init the output layer so d=0 gives zero contribution at init
        nn.init.zeros_(self.step_embed[-1].weight)
        nn.init.zeros_(self.step_embed[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        d: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with step-size conditioning.

        Args:
            x: (B, input_dim) noisy input points.
            t: (B,) timesteps.
            d: (B,) step sizes (defaults to zeros if None).
            y: (B,) class labels (optional).

        Returns:
            (B, input_dim) predicted velocity.
        """
        h = self.input_proj(x)

        # Conditioning: time + step-size [+ class]
        c = self.time_embed(t)
        if d is None:
            d = torch.zeros_like(t)
        c = c + self.step_embed(d)
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
