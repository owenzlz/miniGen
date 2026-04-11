"""
Drifting Model generative process.

Reference: Deng et al., "Generative Modeling via Drifting", 2026.

Core idea: evolve the pushforward distribution during training via a drifting
field V that attracts generated samples toward real data and repels them from
other generated samples. At equilibrium (q = p), V = 0 everywhere.

Training loss: MSE(gen, stopgrad(gen + V))
Inference: single forward pass (1 NFE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftingModel:
    """Drifting Model generative process.

    Uses a mean-shift drifting field with exponential kernel to guide
    generated samples toward the data distribution during training.

    Args:
        latent_dim: Latent noise dimensionality for the generator.
        temp: Temperature for the exponential kernel k(x,y) = exp(-||x-y||/temp).
    """

    def __init__(self, latent_dim: int = 32, temp: float = 0.05):
        self.latent_dim = latent_dim
        self.temp = temp

    def compute_drift(self, gen: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Compute the mean-shift drifting field V.

        V(x) = V_p^+(x) - V_q^-(x), where positive samples (data) attract
        and negative samples (generated) repel, using a batch-normalized
        exponential kernel.

        Args:
            gen: Generated samples [G, D].
            pos: Data (positive) samples [P, D].

        Returns:
            Drift vectors [G, D].
        """
        targets = torch.cat([gen, pos], dim=0)
        G = gen.shape[0]

        # Pairwise distances from generated to all (gen + pos)
        dist = torch.cdist(gen, targets)
        # Mask self-distances to avoid self-attraction
        dist[:, :G].fill_diagonal_(1e6)
        # Exponential kernel
        kernel = (-dist / self.temp).exp()

        # Batch-normalize along both dims (slightly improves performance per paper)
        normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
        normalizer = normalizer.clamp_min(1e-12).sqrt()
        normalized_kernel = kernel / normalizer

        # Positive drift (attraction toward data)
        pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
        pos_V = pos_coeff @ targets[G:]

        # Negative drift (repulsion from other generated samples)
        neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
        neg_V = neg_coeff @ targets[:G]

        return pos_V - neg_V

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy timesteps (drifting model doesn't use timesteps)."""
        return torch.zeros(batch_size, device=device)

    def training_loss(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor, y=None, **kwargs) -> torch.Tensor:
        """Compute drifting loss: MSE(gen, stopgrad(gen + V)).

        Args:
            model: Generator network (noise -> data).
            x0: Real data samples [B, D] (used as positive samples).
            t: Dummy timesteps (ignored).
            y: Class labels (ignored for now).

        Returns:
            Scalar loss.
        """
        device = x0.device
        z = torch.randn(x0.shape[0], self.latent_dim, device=device)
        gen = model(z)

        with torch.no_grad():
            V = self.compute_drift(gen, x0)
            target = (gen + V).detach()

        return F.mse_loss(gen, target)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples (single forward pass).

        Args:
            model: Generator network.
            shape: (num_samples, data_dim).
            device: Device.

        Returns:
            Generated samples (num_samples, data_dim).
        """
        model.eval()
        num_samples = shape[0]
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return model(z)
