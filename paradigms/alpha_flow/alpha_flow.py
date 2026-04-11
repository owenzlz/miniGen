"""
AlphaFlow generative process.

Unifies flow matching and MeanFlow via a tunable alpha parameter that controls
the balance between continuous (JVP-based) and discrete (stepping-based) mean
velocity targets.

Key equations:
    Forward:      z_t = (1-t)*x + t*eps
    True velocity: v = eps - x
    FM target (h=0):           u_tgt = v
    Continuous MF target (h>0): u_tgt = v - h * sg(du/dt)  via JVP
    Discrete MF target (h>0):   u_tgt = (dt*v + (h-dt)*u_next) / h
        where u_next = model(z_t - dt*v, t-dt, h-dt)

    Final target = (1-alpha)*continuous + alpha*discrete  (for MF samples)
    Loss: adaptive weighted MSE

Reference: Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models"
"""
import torch


class AlphaFlow:
    """
    AlphaFlow generative process.

    Args:
        alpha: Balance between continuous (0) and discrete (1) MF targets.
        discrete_dt: Step size for discrete target computation.
        data_proportion: Fraction of batch with r = t (flow matching regime).
        logit_normal_mean: Mean of logit-normal distribution for time sampling.
        logit_normal_std: Std of logit-normal distribution for time sampling.
        norm_p: Exponent for adaptive weight normalization.
        norm_eps: Epsilon for adaptive weight normalization.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        discrete_dt: float = 0.01,
        data_proportion: float = 0.75,
        logit_normal_mean: float = -0.4,
        logit_normal_std: float = 1.0,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
    ):
        self.alpha = alpha
        self.discrete_dt = discrete_dt
        self.data_proportion = data_proportion
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return dummy zeros — t/r are sampled internally in training_loss."""
        return torch.zeros(batch_size, device=device)

    def _adaptive_weight(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """Adaptive weight normalization to prevent high-loss samples from dominating."""
        weight = (per_sample_loss + self.norm_eps) ** self.norm_p
        return per_sample_loss / weight.detach()

    def training_loss(self, model, x0, t_dummy, **kwargs) -> torch.Tensor:
        """
        Compute the AlphaFlow training loss.

        Args:
            model: Single-head model returning u. Signature: model(x, t, h).
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

        # --- Construct noisy samples and target velocity ---
        eps = torch.randn_like(x0)
        z_t = (1 - t[:, None]) * x0 + t[:, None] * eps
        v = eps - x0  # true instantaneous velocity
        h = (t - r).clamp(0.0, 1.0)

        # --- JVP to compute u (prediction) and du/dt (for continuous target) ---
        def u_fn(z_t_in, t_in, r_in):
            return model(z_t_in, t_in, t_in - r_in)

        dtdt = torch.ones(B, device=device)
        drdt = torch.zeros(B, device=device)
        u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v, dtdt, drdt))

        # Continuous target: u_tgt = v - h * sg(du/dt)
        u_tgt = v - h[:, None] * du_dt.detach()

        # --- Discrete target via Euler stepping (if alpha > 0) ---
        if self.alpha > 0:
            dt = torch.full((B,), self.discrete_dt, device=device)
            dt = torch.min(dt, h)  # dt cannot exceed h

            with torch.no_grad():
                z_stepped = z_t - dt[:, None] * v       # z_{t-dt}
                t_stepped = t - dt                       # t - dt
                h_stepped = (h - dt).clamp(min=0)        # (t-dt) - r
                u_next = model(z_stepped, t_stepped, h_stepped)

            # Discrete target: (dt*v + (h-dt)*u_next) / h
            safe_h = h.clamp(min=1e-6)[:, None]
            dt_2d = dt[:, None]
            u_tgt_disc = (dt_2d * v + (h[:, None] - dt_2d) * u_next) / safe_h

            # Blend: only for MF samples (h > 0)
            mf_blend = (self.alpha * (h > 1e-6).float())[:, None]
            u_tgt = u_tgt + mf_blend * (u_tgt_disc.detach() - u_tgt)

        u_tgt = u_tgt.detach()

        # --- Per-sample loss with adaptive weighting ---
        loss = (u - u_tgt).pow(2).sum(dim=-1)  # (B,)
        loss = self._adaptive_weight(loss)
        return loss.mean()

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

        For 1-step: x = z - u(z, t=1, h=1).
        For multi-step: Euler with evenly spaced time steps from 1 to 0.

        Args:
            model: Single-head model returning u. Signature: model(x, t, h).
            shape: (N, D) shape of samples to generate.
            device: Torch device.
            num_steps: Number of sampling steps (1 for exact 1-NFE).

        Returns:
            (N, D) generated samples.
        """
        z = torch.randn(shape, device=device)
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_val = t_steps[i]
            r_val = t_steps[i + 1]
            h_val = t_val - r_val

            B = z.shape[0]
            t_vec = torch.full((B,), t_val.item(), device=device)
            h_vec = torch.full((B,), h_val.item(), device=device)
            u = model(z, t_vec, h_vec)
            z = z - h_val * u

        return z
