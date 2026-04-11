"""
Flow Matching module for generative modeling.

This package provides modular, extensible components for flow matching models:

- **Base classes**: `FlowMatchingBase`, `Interpolant`, `ODESolver`
- **Interpolants**: `LinearInterpolant` (optimal transport path)
- **Solvers**: `EulerSolver`, `HeunSolver`
- **Main class**: `FlowMatching`

Example usage:
    from paradigms.flow_matching import FlowMatching, LinearInterpolant, EulerSolver

    # Default: linear interpolant + Euler solver
    flow = FlowMatching()

    # Training
    t = flow.sample_timesteps(batch_size, device)
    loss = flow.training_loss(model, x0, t, y=labels)

    # Sampling
    samples = flow.sample(model, shape, device, num_steps=50)

Comparison with DDPM:
    - DDPM: discrete timesteps, ancestral sampling, variance schedule
    - Flow: continuous time, ODE solving, interpolant-based
"""

# Base class (imports Interpolant and ODESolver internally)
from .base import FlowMatchingBase

# Base classes from submodules
from .interpolant import Interpolant
from .solver import ODESolver

# Main class
from .flow_matching import FlowMatching

# Interpolants
from .interpolant import LinearInterpolant

# Solvers
from .solver import EulerSolver, HeunSolver

__all__ = [
    # Base
    "FlowMatchingBase",
    "Interpolant",
    "ODESolver",
    # Main
    "FlowMatching",
    # Interpolants
    "LinearInterpolant",
    # Solvers
    "EulerSolver",
    "HeunSolver",
]
