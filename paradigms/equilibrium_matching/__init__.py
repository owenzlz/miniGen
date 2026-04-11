"""
Equilibrium Matching (EqM) module for generative modeling.

Reference:
- Wang & Du, "Equilibrium Matching", 2024.

EqM learns an equilibrium gradient field instead of a time-dependent velocity.
Training uses flow matching interpolation with a modified target (velocity *= ct),
but the model receives t=0 always (time-unconditional). Sampling uses gradient
descent (not ODE integration) on the learned landscape.

Example usage:
    from paradigms.equilibrium_matching import EquilibriumMatching

    eqm = EquilibriumMatching(sampler="ngd", stepsize=0.01)

    # Training
    t = eqm.sample_timesteps(batch_size, device)
    loss = eqm.training_loss(model, x0, t)

    # Sampling
    samples = eqm.sample(model, shape, device, num_steps=250)
"""

from .equilibrium_matching import EquilibriumMatching

__all__ = ["EquilibriumMatching"]
