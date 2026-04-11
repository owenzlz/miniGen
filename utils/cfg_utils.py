"""
Classifier-Free Guidance (CFG) utilities.

CFG is a guidance technique that works with any sampler/solver by modifying
the model outputs at inference time.

Reference: Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop.

================================================================================
CFG FORMULATION
================================================================================

During training:
    - With probability p, replace class label y with null token ∅
    - Model learns both p(x|y) and p(x) (unconditional)

During inference:
    - Compute both conditional and unconditional outputs
    - Combine: output_cfg = output_uncond + scale * (output_cond - output_uncond)

For diffusion (ε-prediction):
    ε_cfg = ε_uncond + s * (ε_cond - ε_uncond)

For flow matching (velocity prediction):
    v_cfg = v_uncond + s * (v_cond - v_uncond)

When scale = 1.0: no guidance (just conditional)
When scale > 1.0: amplifies class-conditional signal
When scale = 0.0: unconditional generation
"""
import torch
import torch.nn as nn
from typing import Optional


class CFGWrapper(nn.Module):
    """Wraps a model to apply classifier-free guidance.

    This wrapper is transparent to any sampler - just pass it instead of
    the original model, and CFG is automatically applied.

    Example:
        # Without CFG
        samples = sampler.sample(model, shape, device, y=labels)

        # With CFG
        cfg_model = CFGWrapper(model, cfg_scale=7.5, null_class=num_classes)
        samples = sampler.sample(cfg_model, shape, device, y=labels)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
    ):
        """Initialize CFG wrapper.

        Args:
            model: The underlying model (predicts ε, v, or x_0).
            cfg_scale: Guidance scale s. 1.0 = no guidance, >1.0 = stronger guidance.
            null_class: Class index for unconditional. Default: model.num_classes.
        """
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

        # Determine null class
        if null_class is None:
            null_class = getattr(model, 'num_classes', 0)
        self.null_class = null_class

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with CFG applied.

        Args:
            x: Input tensor (noisy sample x_t).
            t: Timestep tensor.
            y: Class labels. Required for CFG.
            **kwargs: Additional model arguments.

        Returns:
            Model output with CFG applied.
        """
        # No guidance if scale is 1.0 or no labels
        if self.cfg_scale == 1.0 or y is None:
            return self.model(x, t, y=y, **kwargs)

        # Unconditional prediction (null class)
        y_null = torch.full_like(y, self.null_class)
        out_uncond = self.model(x, t, y=y_null, **kwargs)

        # Conditional prediction
        out_cond = self.model(x, t, y=y, **kwargs)

        # CFG: out = out_uncond + scale * (out_cond - out_uncond)
        return out_uncond + self.cfg_scale * (out_cond - out_uncond)

    # Delegate attribute access to wrapped model
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class CFGWrapperBatched(nn.Module):
    """Batched CFG wrapper - runs conditional and unconditional in one forward pass.

    More efficient than CFGWrapper when batch size is small, as it avoids
    two separate forward passes. However, uses 2x memory.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

        if null_class is None:
            null_class = getattr(model, 'num_classes', 0)
        self.null_class = null_class

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if self.cfg_scale == 1.0 or y is None:
            return self.model(x, t, y=y, **kwargs)

        # Batch both conditional and unconditional
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        y_null = torch.full_like(y, self.null_class)
        y_double = torch.cat([y, y_null], dim=0)

        # Single forward pass
        out_double = self.model(x_double, t_double, y=y_double, **kwargs)

        # Split and apply CFG
        out_cond, out_uncond = out_double.chunk(2, dim=0)
        return out_uncond + self.cfg_scale * (out_cond - out_uncond)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
