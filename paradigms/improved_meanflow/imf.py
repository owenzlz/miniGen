"""
Improved MeanFlow (iMF) generative process.

Learns an average velocity field u(z_t, r, t) instead of the instantaneous
velocity, enabling exact 1-step generation. The "improved" version reformulates
the loss so the regression target is network-independent (v_target = eps - x),
making training stable.

Note: For 2D data we condition on both t and h = t - r (unlike the reference
which conditions on h only), because low-dimensional z_t does not carry enough
information to infer the noise level.

Reference: Geng et al., "Improved Mean Flows", arXiv:2512.02012
"""
import torch
import torch.nn.functional as F


class ImprovedMeanFlow:
    """
    Improved MeanFlow generative process.

    Training loss uses JVP to compute du/dt, then constructs the compound
    function V = u + (t-r) * stopgrad(du/dt) and regresses toward v_target = eps - x.
    An auxiliary v-head loss is added.

    Time sampling uses logit-normal distribution for t and r, with a fraction
    of the batch using r = t (flow matching regime).

    Args:
        data_proportion: Fraction of batch with r = t (flow matching regime).
        logit_normal_mean: Mean of logit-normal distribution for time sampling.
        logit_normal_std: Std of logit-normal distribution for time sampling.
        norm_p: Exponent for adaptive weight normalization (1.0 = reference default).
        norm_eps: Epsilon for adaptive weight normalization.
    """

    def __init__(
        self,
        data_proportion: float = 0.5,
        logit_normal_mean: float = -0.4,
        logit_normal_std: float = 1.0,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
    ):
        self.data_proportion = data_proportion
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy zeros — t/r are sampled internally in training_loss."""
        return torch.zeros(batch_size, device=device)

    def _adaptive_weight(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """Adaptive weight normalization (reference: imf.py adp_wt_fn).

        Normalizes per-sample loss to prevent high-loss samples from dominating.
        With norm_p=1: loss / sg(loss + eps) ≈ 1 for large loss, ≈ loss/eps for small.
        """
        weight = (per_sample_loss + self.norm_eps) ** self.norm_p
        return per_sample_loss / weight.detach()

    def training_loss(self, model, x0, t_dummy, **kwargs) -> torch.Tensor:
        """
        Compute the improved MeanFlow training loss.

        Args:
            model: Dual-head model returning (u, v). Signature: model(x, t, h).
            x0: (B, D) clean data samples.
            t_dummy: Unused (timesteps sampled internally).

        Returns:
            Scalar loss.
        """
        B, D = x0.shape
        device = x0.device

        # --- Sample t and r from logit-normal ---
        t = torch.sigmoid(
            self.logit_normal_mean
            + self.logit_normal_std * torch.randn(B, device=device)
        )
        r = torch.sigmoid(
            self.logit_normal_mean
            + self.logit_normal_std * torch.randn(B, device=device)
        )
        # Enforce t >= r
        t, r = torch.max(t, r), torch.min(t, r)

        # FM regime: first data_proportion fraction has r = t
        data_size = int(B * self.data_proportion)
        r[:data_size] = t[:data_size]

        # --- Construct noisy samples and target ---
        eps = torch.randn_like(x0)
        z_t = (1 - t[:, None]) * x0 + t[:, None] * eps
        v_target = eps - x0  # network-independent target

        # --- Get v_c: instantaneous velocity from v-head at h=0 (detached) ---
        with torch.no_grad():
            h_zero = torch.zeros(B, device=device)
            _, v_c = model(z_t, t, h_zero)

        # --- JVP to compute du/dt ---
        # u_fn takes (z_t, t, r) and returns (u, v) where v is aux
        def u_fn(z_t_in, t_in, r_in):
            h = t_in - r_in
            u, v = model(z_t_in, t_in, h)
            return u, v  # u is primal output, v is aux

        dtdt = torch.ones(B, device=device)
        dtdr = torch.zeros(B, device=device)

        u, du_dt, v = torch.func.jvp(
            u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True
        )

        # --- Compound function V = u + (t-r) * sg(du/dt) ---
        V = u + (t - r)[:, None] * du_dt.detach()

        # --- Per-sample loss with adaptive weighting ---
        # Sum over features (D), then adaptive weight, then mean over batch
        loss_u = (V - v_target).pow(2).sum(dim=-1)  # (B,)
        loss_u = self._adaptive_weight(loss_u)

        loss_v = (v - v_target).pow(2).sum(dim=-1)  # (B,)
        loss_v = self._adaptive_weight(loss_v)

        loss = (loss_u + loss_v).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        device,
        num_steps: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples via Euler integration of the average velocity field.

        For 1-step (num_steps=1): x = z - u(z, t=1, h=1).
        For multi-step: Euler with evenly spaced time steps from 1 to 0.

        Args:
            model: Dual-head model returning (u, v). Signature: model(x, t, h).
            shape: (N, D) shape of samples to generate.
            device: Torch device.
            num_steps: Number of sampling steps (1 for exact 1-NFE).

        Returns:
            (N, D) generated samples.
        """
        z = torch.randn(shape, device=device)
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            h = t - r

            B = z.shape[0]
            t_vec = torch.full((B,), t.item(), device=device)
            h_vec = torch.full((B,), h.item(), device=device)
            u, _ = model(z, t_vec, h_vec)
            z = z - h * u

        return z
