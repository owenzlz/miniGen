"""
Equilibrium Matching (EqM) implementation.

Reference:
- Wang & Du, "Equilibrium Matching", 2024.

Key ideas:
- Training uses flow matching interpolation but scales the velocity target by a
  piecewise-linear coefficient ct (4.0 for t<0.8, linearly decays to 0 at t=1).
- The model always receives t=0 (time-unconditional), learning a single
  equilibrium gradient field rather than a time-dependent velocity.
- Sampling uses gradient descent (GD), Nesterov accelerated GD (NAG-GD),
  or Langevin dynamics on the learned landscape.

EBM mode (ebm="l2"):
- The model output v is used to define energy E = -||v||^2 / 2
- The drift becomes f(x) = ∇_x E = -(∂v/∂x)^T v (conservative field)
- Conservative fields can support multiple equilibrium basins, unlike
  the direct velocity field which averages to a single attractor in low-D.

EBM mode (ebm="dot"):
- Energy E = v(x) · x (dot product of model output with input)
- The drift becomes f(x) = ∇_x E = v + (∂v/∂x)^T x
- Includes the direct velocity v as a component, plus a conservative correction.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ebm_l2_drift(model, x, t_zero, uncond, **kwargs):
    """Compute conservative drift via L2 energy: E = -||model(x)||^2 / 2.

    Returns ∇_x E, which is a curl-free (conservative) vector field.
    Uses autograd to compute the gradient through the model.
    """
    x_in = x.detach().requires_grad_(True)
    v = model(x_in, t_zero, uncond=uncond, **kwargs)
    energy = -(v ** 2).sum(dim=-1) / 2  # (B,)
    grad = torch.autograd.grad(
        energy.sum(), x_in, create_graph=model.training
    )[0]
    return grad


def _ebm_dot_drift(model, x, t_zero, uncond, **kwargs):
    """Compute conservative drift via dot energy: E = model(x) · x.

    Returns ∇_x E = v + (∂v/∂x)^T x, which includes the direct
    velocity v as a component plus a conservative correction.
    """
    x_in = x.detach().requires_grad_(True)
    v = model(x_in, t_zero, uncond=uncond, **kwargs)
    energy = (v * x_in).sum(dim=-1)  # (B,)
    grad = torch.autograd.grad(
        energy.sum(), x_in, create_graph=model.training
    )[0]
    return grad


class EquilibriumMatching:
    """Equilibrium Matching generative model.

    Learns an equilibrium gradient field by training with flow matching
    interpolation and a time-varying scaling coefficient, but always
    passing t=0 to the model so it learns a single stationary field.

    Training:
        1. Sample x_0 ~ p_data, eps ~ N(0, I), t ~ U[0, 1]
        2. Compute x_t = (1-t)*eps + t*x_0  (linear interpolation)
        3. Compute velocity target = (x_0 - eps) * ct(t)
        4. Model receives (x_t, t=0) — no time conditioning
        5. Minimize ||model(x_t, 0) - velocity_target||^2

    Sampling (GD):
        x_{k+1} = x_k + stepsize * model(x_k, 0)

    Sampling (NGD):
        x' = x_k + stepsize * mu * m_k
        m_{k+1} = model(x', 0)
        x_{k+1} = x_k + stepsize * m_{k+1}

    Sampling (Langevin):
        x_{k+1} = x_k + stepsize * model(x_k, 0) + sqrt(2 * stepsize * noise_scale) * z
        With optional annealing: noise_scale decays from noise_scale to noise_end.
    """

    def __init__(
        self,
        sampler: str = "langevin",
        stepsize: float = 0.0017,
        num_sampling_steps: int = 1000,
        mu: float = 0.3,
        noise_scale: float = 2.0,
        noise_end: float = 0.01,
        langevin_anneal: bool = True,
        uncond: bool = False,
        ebm: str = "none",
        ct_floor: float = 0.0,
    ):
        """Initialize Equilibrium Matching.

        Args:
            sampler: Sampling method — "gd", "ngd" (Nesterov), "langevin", or "svgd".
            stepsize: Step size for sampling.
            num_sampling_steps: Default number of sampling steps.
            mu: Momentum coefficient for NAG-GD (only used if sampler="ngd").
            noise_scale: Langevin noise scale (only used if sampler="langevin").
            noise_end: End noise scale for annealed Langevin (only if langevin_anneal=True).
            langevin_anneal: Whether to anneal noise from noise_scale to noise_end.
            uncond: If True, zero out time conditioning entirely (reference behavior).
                    If False, pass t=0 through sinusoidal embedding (produces fixed non-zero conditioning).
            ebm: Energy-based model formulation. "none" for direct velocity,
                 "l2" for E=-||v||^2/2 with conservative drift ∇_x E.
            ct_floor: Minimum value for ct coefficient. Default 0.0 (standard schedule).
                      Set to e.g. 4.0 for constant ct, preserving local structure near data.
        """
        self.sampler = sampler
        self.stepsize = stepsize
        self.num_sampling_steps = num_sampling_steps
        self.mu = mu
        self.noise_scale = noise_scale
        self.noise_end = noise_end
        self.langevin_anneal = langevin_anneal
        self.uncond = uncond
        self.ebm = ebm
        self.ct_floor = ct_floor

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, 1] for training interpolation.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1].
        """
        return torch.rand(batch_size, device=device)

    def compute_ct(self, t: torch.Tensor) -> torch.Tensor:
        """Compute piecewise-linear scaling coefficient ct.

        ct = 4.0 for t < 0.8, then linearly decays to 0 at t = 1.
        Formally: ct = min(1, (1-t)/(1-interp)) * 4.0 with interp=0.8.

        Args:
            t: Time values, shape (B,).

        Returns:
            Scaling coefficients, shape (B,).
        """
        interp = 0.8
        ct = torch.minimum(
            torch.ones_like(t),
            1.0 / (1.0 - interp) - t / (1.0 - interp),
        ) * 4.0
        if self.ct_floor > 0:
            ct = ct.clamp(min=self.ct_floor)
        return ct

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute EqM training loss.

        Standard mode (ebm="none"):
            L = E_{t, x_0, eps} ||model(x_t, 0) - (x_0 - eps) * ct(t)||^2

        EBM mode (ebm="l2"):
            v = model(x_t, 0), E = -||v||^2/2, f = ∇_x E
            L = E_{t, x_0, eps} ||f(x_t) - (x_0 - eps) * ct(t)||^2
            Gradients flow through autograd for conservative field training.

        Args:
            model: Network that predicts the equilibrium gradient field.
            x0: Clean data samples, shape (B, D).
            t: Time values in [0, 1], shape (B,).
            **kwargs: Additional model arguments (e.g., y for class labels).

        Returns:
            Scalar MSE loss.
        """
        noise = torch.randn_like(x0)
        t_expand = t.unsqueeze(-1)

        # Linear interpolation: x_t = (1-t)*noise + t*x0
        xt = (1 - t_expand) * noise + t_expand * x0

        # Velocity target scaled by ct
        velocity_target = (x0 - noise) * self.compute_ct(t).unsqueeze(-1)

        # Model sees t=0 (time-unconditional)
        t_zero = torch.zeros_like(t)

        if self.ebm == "l2":
            velocity_pred = _ebm_l2_drift(model, xt, t_zero, self.uncond, **kwargs)
        elif self.ebm == "dot":
            velocity_pred = _ebm_dot_drift(model, xt, t_zero, self.uncond, **kwargs)
        else:
            velocity_pred = model(xt, t_zero, uncond=self.uncond, **kwargs)

        return F.mse_loss(velocity_pred, velocity_target)

    def _get_drift(self, model, x, t_zero, **kwargs):
        """Get the drift vector at x, using EBM formulation if configured."""
        if self.ebm == "l2":
            with torch.enable_grad():
                return _ebm_l2_drift(model, x, t_zero, self.uncond, **kwargs)
        elif self.ebm == "dot":
            with torch.enable_grad():
                return _ebm_dot_drift(model, x, t_zero, self.uncond, **kwargs)
        else:
            return model(x, t_zero, uncond=self.uncond, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using GD, NGD, or Langevin dynamics on the learned field.

        Args:
            model: Trained equilibrium gradient field model.
            shape: Shape of samples to generate (batch_size, D).
            device: Device to generate on.
            num_steps: Number of steps (default: self.num_sampling_steps).
            **kwargs: Additional model arguments.

        Returns:
            Generated samples.
        """
        model.eval()

        x = torch.randn(shape, device=device)
        t_zero = torch.zeros(shape[0], device=device)
        steps = num_steps or self.num_sampling_steps

        if self.sampler == "ngd":
            # Nesterov accelerated gradient descent
            m = torch.zeros_like(x)
            for _ in range(steps):
                x_lookahead = x + self.stepsize * m * self.mu
                m = self._get_drift(model, x_lookahead, t_zero, **kwargs)
                x = x + m * self.stepsize

        elif self.sampler == "langevin":
            # Langevin dynamics with optional annealing
            for k in range(steps):
                out = self._get_drift(model, x, t_zero, **kwargs)
                if self.langevin_anneal:
                    frac = k / max(steps - 1, 1)
                    ns = self.noise_scale * (1 - frac) + self.noise_end * frac
                else:
                    ns = self.noise_scale
                noise = torch.randn_like(x) * math.sqrt(2 * self.stepsize * ns)
                x = x + self.stepsize * out + noise

        elif self.sampler == "svgd":
            # Stein Variational Gradient Descent — adds inter-particle
            # repulsion via RBF kernel to prevent mode collapse.
            N = shape[0]
            for k in range(steps):
                out = self._get_drift(model, x, t_zero, **kwargs)  # (N, D)
                # Pairwise squared distances
                diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, D)
                sq_dist = (diff ** 2).sum(dim=-1)  # (N, N)
                # Median heuristic for kernel bandwidth
                h_sq = sq_dist.median() / math.log(max(N, 2))
                h_sq = h_sq.clamp(min=1e-6)
                # RBF kernel
                K = torch.exp(-sq_dist / (2 * h_sq))  # (N, N)
                # Kernel gradient: ∇_xj k(x_j, x_i) = k * (x_i - x_j) / h²
                grad_K = K.unsqueeze(-1) * diff / h_sq  # (N, N, D)
                # SVGD update: 1/N * Σ_j [k(x_j,x_i)*f(x_j) + ∇_xj k(x_j,x_i)]
                drive = K @ out  # (N, N) @ (N, D) = (N, D)
                repulse = grad_K.sum(dim=1)  # (N, D), sum over j
                update = (drive + repulse) / N
                # Optional Langevin noise
                if self.noise_scale > 0:
                    if self.langevin_anneal:
                        frac = k / max(steps - 1, 1)
                        ns = self.noise_scale * (1 - frac) + self.noise_end * frac
                    else:
                        ns = self.noise_scale
                    noise = torch.randn_like(x) * math.sqrt(2 * self.stepsize * ns)
                else:
                    noise = 0.0
                x = x + self.stepsize * update + noise

        else:
            # Standard gradient descent
            for _ in range(steps):
                out = self._get_drift(model, x, t_zero, **kwargs)
                x = x + out * self.stepsize

        return x
