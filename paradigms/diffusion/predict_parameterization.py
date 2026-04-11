"""
Pure functions for prediction parameterization conversions.

All functions are stateless and receive schedule values as arguments,
making them easy to test and reuse across different diffusion implementations.
"""
from typing import Tuple
import torch

from .base import Parameterization
from .utils import extract


def predict_x0_from_eps(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Predict x0 from noise prediction.

    x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

    Args:
        xt: Noisy sample at timestep t.
        t: Timestep indices.
        eps: Predicted noise.
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Predicted clean sample x0.
    """
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, xt.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, xt.shape)
    return (xt - sqrt_one_minus_alpha * eps) / sqrt_alpha


def predict_eps_from_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    x0: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Predict noise from x0 prediction.

    eps = (x_t - sqrt(alpha_bar_t) * x_0) / sqrt(1 - alpha_bar_t)

    Args:
        xt: Noisy sample at timestep t.
        t: Timestep indices.
        x0: Predicted clean sample.
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Predicted noise.
    """
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, xt.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, xt.shape)
    return (xt - sqrt_alpha * x0) / sqrt_one_minus_alpha


def predict_x0_from_v(
    xt: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Predict x0 from v-prediction.

    x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v

    Args:
        xt: Noisy sample at timestep t.
        t: Timestep indices.
        v: Predicted velocity.
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Predicted clean sample x0.
    """
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, xt.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, xt.shape)
    return sqrt_alpha * xt - sqrt_one_minus_alpha * v


def predict_v_from_x0_eps(
    x0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Compute v-prediction target from x0 and noise.

    v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0

    Args:
        x0: Clean sample.
        eps: Noise.
        t: Timestep indices.
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Velocity target v.
    """
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_alpha * eps - sqrt_one_minus_alpha * x0


def get_v(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Alias for predict_v_from_x0_eps for backward compatibility."""
    return predict_v_from_x0_eps(x0, noise, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)


def model_output_to_x0(
    model_output: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    parameterization: Parameterization,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Convert model output to x0 prediction based on parameterization.

    Args:
        model_output: Raw model output.
        xt: Noisy sample at timestep t.
        t: Timestep indices.
        parameterization: What the model predicts (EPS, X0, or V).
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Predicted clean sample x0.
    """
    if parameterization == Parameterization.EPS:
        return predict_x0_from_eps(xt, t, model_output, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    elif parameterization == Parameterization.X0:
        return model_output
    elif parameterization == Parameterization.V:
        return predict_x0_from_v(xt, t, model_output, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")


def get_training_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    parameterization: Parameterization,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Get training target based on parameterization.

    Args:
        x0: Clean sample.
        noise: Added noise.
        t: Timestep indices.
        parameterization: What the model should predict (EPS, X0, or V).
        sqrt_alphas_cumprod: sqrt(alpha_bar) schedule values.
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar) schedule values.

    Returns:
        Training target tensor.
    """
    if parameterization == Parameterization.EPS:
        return noise
    elif parameterization == Parameterization.X0:
        return x0
    elif parameterization == Parameterization.V:
        return get_v(x0, noise, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
