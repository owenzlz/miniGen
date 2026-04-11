"""
Autoregressive generative process for 2D points.

Reference: The autoregressive factorization p(x) = prod_d p(x_d | x_{<d})
is the foundation of GPT, PixelCNN, and modern AR image generators.

Training: Teacher forcing with cross-entropy on discretized coordinates.
Sampling: Sequential generation dimension by dimension.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveGeneration:
    """Autoregressive generative process.

    Wraps an AR model to provide the standard process interface
    (sample_timesteps, training_loss, sample) used by train.py.

    Coordinates are discretized into uniform bins over [-1, 1].
    The model predicts a categorical distribution over bins for each dimension,
    conditioned on all previously generated dimensions.

    Args:
        num_bins: Number of discretization bins per dimension.
        temperature: Sampling temperature (lower = sharper, higher = more diverse).
    """

    def __init__(self, num_bins: int = 128, temperature: float = 1.0):
        self.num_bins = num_bins
        self.temperature = temperature

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous [-1, 1] to discrete [0, num_bins-1]."""
        return ((x.clamp(-1, 1) + 1) / 2 * (self.num_bins - 1)).round().long()

    def undiscretize(self, bins: torch.Tensor) -> torch.Tensor:
        """Map discrete [0, num_bins-1] to continuous [-1, 1]."""
        return bins.float() / (self.num_bins - 1) * 2 - 1

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (AR doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(
        self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute cross-entropy loss with teacher forcing.

        Args:
            model: ARTransformer2D instance.
            x0: Clean data samples (B, D) in [-1, 1].
            t: Ignored (AR doesn't use timesteps).
        """
        y = kwargs.get("y")
        x_bins = self.discretize(x0)  # (B, D) long
        logits = model(x_bins, y=y)   # (B, D, num_bins)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_bins),
            x_bins.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples autoregressively, dimension by dimension.

        Args:
            model: ARTransformer2D instance.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Generated samples (num_samples, data_dim) in [-1, 1].
        """
        model.eval()
        B, D = shape
        y = kwargs.get("y")

        x_bins = torch.zeros(B, D, device=device, dtype=torch.long)

        for d in range(D):
            logits = model(x_bins, y=y)  # (B, D, num_bins)
            probs = F.softmax(logits[:, d] / self.temperature, dim=-1)
            x_bins[:, d] = torch.multinomial(probs, 1).squeeze(-1)

        return self.undiscretize(x_bins)
