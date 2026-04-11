"""
Tiny causal transformer for autoregressive 2D generation.

A minimal GPT-style architecture that factorizes p(x1, x2) = p(x1) * p(x2|x1)
using discretized coordinates. Each dimension is treated as a token in a
length-2 sequence with causal masking.

Architecture:
    Input: [START, embed(x1)]  (shifted input, length = input_dim)
    Causal self-attention ensures each position only sees previous positions.
    Output: [logits_x1, logits_x2]  (categorical over num_bins per dimension)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, T, T)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention and FFN."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ARTransformer2D(nn.Module):
    """
    Tiny causal transformer for autoregressive 2D generation.

    Factorizes p(x1, x2) = p(x1) * p(x2|x1) by treating each coordinate
    as a discrete token (quantized into num_bins bins over [-1, 1]).

    The input sequence is shifted: [START, embed(x1)] produces logits for [x1, x2].
    Causal masking ensures the prediction for x_d only depends on x_{<d}.

    Args:
        num_bins: Number of discretization bins per coordinate.
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        input_dim: Data dimensionality (sequence length). Always 2 for this playground.
        num_classes: Number of classes for conditional generation (0 = unconditional).
        class_dropout_prob: Probability of dropping class labels for CFG training.
    """

    def __init__(
        self,
        num_bins: int = 128,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        input_dim: int = 2,
        num_classes: int = 0,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Token embedding: bin index -> hidden_size
        self.token_embed = nn.Embedding(num_bins, hidden_size)

        # Learnable start token
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

        # Positional embedding
        self.pos_embed = nn.Embedding(input_dim, hidden_size)

        # Class embedding (optional)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes + 1, hidden_size)
        else:
            self.class_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_bins)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (GPT-2 style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Small init for start token
        nn.init.normal_(self.start_token, std=0.02)

    def _drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Randomly drop labels for classifier-free guidance training."""
        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
            labels = torch.where(drop_mask, self.num_classes, labels)
        return labels

    def forward(
        self,
        x_bins: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_bins: (B, D) discrete bin indices in [0, num_bins-1].
                    During training: ground truth bins (teacher forcing).
                    During sampling: partially filled by the process.
            y: (B,) class labels (optional).

        Returns:
            (B, D, num_bins) logits for each dimension.
        """
        B, D = x_bins.shape

        # Build shifted input: [START, embed(x_1), ..., embed(x_{D-1})]
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, H)
        if D > 1:
            tok = self.token_embed(x_bins[:, :-1])  # (B, D-1, H)
            h = torch.cat([start, tok], dim=1)       # (B, D, H)
        else:
            h = start  # (B, 1, H)

        # Positional embedding
        positions = torch.arange(D, device=x_bins.device)
        h = h + self.pos_embed(positions)

        # Class conditioning
        if self.class_embed is not None and y is not None:
            y = self._drop_labels(y)
            h = h + self.class_embed(y).unsqueeze(1)

        # Causal transformer
        for block in self.blocks:
            h = block(h)

        # Output logits
        h = self.ln_f(h)
        logits = self.head(h)  # (B, D, num_bins)
        return logits
