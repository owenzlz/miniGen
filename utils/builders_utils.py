"""
Factory functions for creating models, processes, schedules, and samplers.

These functions work with both OmegaConf and dict configs.
"""
from enum import Enum
from typing import Union, Optional

import torch


class SamplerType(str, Enum):
    """Available sampler types for diffusion models."""
    DDPM = "ddpm"
    DDIM = "ddim"


def _get(cfg, *keys, default=None):
    """Get nested value from config (works with both dict and OmegaConf)."""
    value = cfg
    for key in keys:
        if hasattr(value, "get"):
            value = value.get(key)
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
        if value is None:
            return default
    return value


def create_diffusion_schedule(cfg, device: Optional[torch.device] = None):
    """Create noise schedule from config for diffusion models.

    Args:
        cfg: Config object (OmegaConf or dict) with diffusion.schedule section.
        device: Optional device to move schedule to.

    Returns:
        Schedule instance (LinearBetaSchedule or CosineBetaSchedule).
    """
    from paradigms.diffusion import LinearBetaSchedule, CosineBetaSchedule

    schedule_type = _get(cfg, "diffusion", "schedule", "type")
    num_timesteps = _get(cfg, "diffusion", "num_timesteps")

    if schedule_type == "linear":
        schedule = LinearBetaSchedule(
            num_timesteps=num_timesteps,
            beta_start=_get(cfg, "diffusion", "schedule", "beta_start"),
            beta_end=_get(cfg, "diffusion", "schedule", "beta_end"),
        )
    elif schedule_type == "cosine":
        schedule = CosineBetaSchedule(num_timesteps=num_timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    if device is not None:
        schedule = schedule.to(device)

    return schedule


def create_generative_process(cfg, device: torch.device):
    """Create generative process from config.

    Args:
        cfg: Config object (OmegaConf or dict).
        device: Torch device.

    Returns:
        Generative process with sample_timesteps(), training_loss(), and sample() methods.
    """
    process_type = _get(cfg, "type", default="ddpm")

    if process_type == "ddpm":
        from paradigms.diffusion import DDPM, Parameterization

        schedule = create_diffusion_schedule(cfg, device)
        param = Parameterization(_get(cfg, "diffusion", "parameterization"))
        return DDPM(schedule, param)

    elif process_type == "flow_matching":
        from paradigms.flow_matching import FlowMatching, EulerSolver, HeunSolver

        solver_type = _get(cfg, "flow", "solver", default="euler")
        if solver_type == "heun":
            solver = HeunSolver()
        else:
            solver = EulerSolver()

        return FlowMatching(solver=solver)

    elif process_type == "variational_autoencoder":
        from paradigms.variational_autoencoder import VAE

        return VAE(
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
            beta=_get(cfg, "variational_autoencoder", "beta", default=1.0),
        )

    elif process_type == "generative_adversarial_network":
        from paradigms.generative_adversarial_network import GAN
        from paradigms.generative_adversarial_network import Discriminator

        disc = Discriminator(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "generative_adversarial_network", "disc_hidden_size", default=256),
            depth=_get(cfg, "generative_adversarial_network", "disc_depth", default=4),
        ).to(device)

        disc_optimizer = torch.optim.Adam(
            disc.parameters(),
            lr=_get(cfg, "generative_adversarial_network", "disc_lr", default=1e-4),
            betas=(0.5, 0.999),
        )

        return GAN(
            discriminator=disc,
            disc_optimizer=disc_optimizer,
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
            n_critic=_get(cfg, "generative_adversarial_network", "n_critic", default=1),
        )

    elif process_type == "score_matching_ncsn":
        from paradigms.score_matching_ncsn import ScoreMatchingNCSN

        return ScoreMatchingNCSN(
            num_noise_levels=_get(cfg, "score_matching_ncsn", "num_noise_levels", default=10),
            sigma_min=_get(cfg, "score_matching_ncsn", "sigma_min", default=0.01),
            sigma_max=_get(cfg, "score_matching_ncsn", "sigma_max", default=1.0),
            langevin_steps=_get(cfg, "score_matching_ncsn", "langevin_steps", default=100),
            langevin_eps=_get(cfg, "score_matching_ncsn", "langevin_eps", default=0.1),
        )

    elif process_type == "score_matching_hyvarinen":
        from paradigms.score_matching_hyvarinen import ScoreMatchingHyvarinen

        return ScoreMatchingHyvarinen(
            langevin_steps=_get(cfg, "score_matching_hyvarinen", "langevin_steps", default=1000),
            langevin_eps=_get(cfg, "score_matching_hyvarinen", "langevin_eps", default=1e-4),
        )

    elif process_type == "normalizing_flow":
        from paradigms.normalizing_flow import NormalizingFlow

        return NormalizingFlow()

    elif process_type == "continuous_normalizing_flow":
        from paradigms.continuous_normalizing_flow import ContinuousNormalizingFlow

        return ContinuousNormalizingFlow(
            num_integration_steps=_get(cfg, "continuous_normalizing_flow", "num_integration_steps", default=10),
        )

    elif process_type == "consistency_training":
        from paradigms.consistency_training import ConsistencyTraining

        return ConsistencyTraining(
            sigma_min=_get(cfg, "consistency_training", "sigma_min", default=0.002),
            sigma_max=_get(cfg, "consistency_training", "sigma_max", default=1.0),
            sigma_data=_get(cfg, "consistency_training", "sigma_data", default=0.5),
            rho=_get(cfg, "consistency_training", "rho", default=7.0),
            initial_timesteps=_get(cfg, "consistency_training", "initial_timesteps", default=10),
            final_timesteps=_get(cfg, "consistency_training", "final_timesteps", default=150),
            total_training_steps=_get(cfg, "training", "total_steps", default=20000),
            p_mean=_get(cfg, "consistency_training", "p_mean", default=-0.4),
            p_std=_get(cfg, "consistency_training", "p_std", default=1.0),
        )

    elif process_type == "shortcut_model":
        from paradigms.shortcut_model import ShortcutModel

        return ShortcutModel(
            num_steps=_get(cfg, "shortcut_model", "num_steps", default=128),
            bootstrap_every=_get(cfg, "shortcut_model", "bootstrap_every", default=4),
        )

    elif process_type == "energy_based_model":
        from paradigms.energy_based_model import EBM

        return EBM(
            cd_steps=_get(cfg, "energy_based_model", "cd_steps", default=60),
            cd_step_size=_get(cfg, "energy_based_model", "cd_step_size", default=0.01),
            cd_data_init=_get(cfg, "energy_based_model", "cd_data_init", default=False),
            cd_noise_std=_get(cfg, "energy_based_model", "cd_noise_std", default=0.3),
            cd_noise_scale=_get(cfg, "energy_based_model", "cd_noise_scale", default=1.0),
            buffer_size=_get(cfg, "energy_based_model", "buffer_size", default=10000),
            reinit_freq=_get(cfg, "energy_based_model", "reinit_freq", default=0.05),
            energy_reg=_get(cfg, "energy_based_model", "energy_reg", default=0.001),
            langevin_steps=_get(cfg, "energy_based_model", "langevin_steps", default=1000),
            langevin_step_size=_get(cfg, "energy_based_model", "langevin_step_size", default=0.01),
            langevin_grad_clip=_get(cfg, "energy_based_model", "langevin_grad_clip", default=0.0),
            langevin_clamp=_get(cfg, "energy_based_model", "langevin_clamp", default=0.0),
            langevin_noise_scale=_get(cfg, "energy_based_model", "langevin_noise_scale", default=1.0),
            langevin_anneal=_get(cfg, "energy_based_model", "langevin_anneal", default=False),
            langevin_noise_end=_get(cfg, "energy_based_model", "langevin_noise_end", default=0.01),
        )

    elif process_type == "autoencoder":
        from paradigms.autoencoder import AE

        return AE(
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
        )

    elif process_type == "drifting_model":
        from paradigms.drifting_model import DriftingModel

        return DriftingModel(
            latent_dim=_get(cfg, "model", "latent_dim", default=32),
            temp=_get(cfg, "drifting_model", "temp", default=0.05),
        )

    elif process_type == "meanflow":
        from paradigms.meanflow import MeanFlow

        return MeanFlow(
            data_proportion=_get(cfg, "meanflow", "data_proportion", default=0.75),
            logit_normal_mean=_get(cfg, "meanflow", "logit_normal_mean", default=-0.4),
            logit_normal_std=_get(cfg, "meanflow", "logit_normal_std", default=1.0),
            norm_p=_get(cfg, "meanflow", "norm_p", default=1.0),
            norm_eps=_get(cfg, "meanflow", "norm_eps", default=0.01),
        )

    elif process_type == "alpha_flow":
        from paradigms.alpha_flow import AlphaFlow

        return AlphaFlow(
            alpha=_get(cfg, "alpha_flow", "alpha", default=0.0),
            discrete_dt=_get(cfg, "alpha_flow", "discrete_dt", default=0.01),
            data_proportion=_get(cfg, "alpha_flow", "data_proportion", default=0.75),
            logit_normal_mean=_get(cfg, "alpha_flow", "logit_normal_mean", default=-0.4),
            logit_normal_std=_get(cfg, "alpha_flow", "logit_normal_std", default=1.0),
            norm_p=_get(cfg, "alpha_flow", "norm_p", default=1.0),
            norm_eps=_get(cfg, "alpha_flow", "norm_eps", default=0.01),
        )

    elif process_type == "improved_meanflow":
        from paradigms.improved_meanflow import ImprovedMeanFlow

        return ImprovedMeanFlow(
            data_proportion=_get(cfg, "improved_meanflow", "data_proportion", default=0.5),
            logit_normal_mean=_get(cfg, "improved_meanflow", "logit_normal_mean", default=-0.4),
            logit_normal_std=_get(cfg, "improved_meanflow", "logit_normal_std", default=1.0),
            norm_p=_get(cfg, "improved_meanflow", "norm_p", default=1.0),
            norm_eps=_get(cfg, "improved_meanflow", "norm_eps", default=0.01),
        )

    elif process_type == "equilibrium_matching":
        from paradigms.equilibrium_matching import EquilibriumMatching

        return EquilibriumMatching(
            sampler=_get(cfg, "equilibrium_matching", "sampler", default="langevin"),
            stepsize=_get(cfg, "equilibrium_matching", "stepsize", default=0.0017),
            num_sampling_steps=_get(cfg, "equilibrium_matching", "num_sampling_steps", default=1000),
            mu=_get(cfg, "equilibrium_matching", "mu", default=0.3),
            noise_scale=_get(cfg, "equilibrium_matching", "noise_scale", default=2.0),
            noise_end=_get(cfg, "equilibrium_matching", "noise_end", default=0.01),
            langevin_anneal=_get(cfg, "equilibrium_matching", "langevin_anneal", default=True),
            uncond=_get(cfg, "equilibrium_matching", "uncond", default=False),
            ebm=_get(cfg, "equilibrium_matching", "ebm", default="none"),
            ct_floor=_get(cfg, "equilibrium_matching", "ct_floor", default=0.0),
        )

    elif process_type == "autoregressive":
        from paradigms.autoregressive import AutoregressiveGeneration

        return AutoregressiveGeneration(
            num_bins=_get(cfg, "autoregressive", "num_bins", default=128),
            temperature=_get(cfg, "autoregressive", "temperature", default=1.0),
        )

    else:
        raise ValueError(f"Unknown process type: {process_type}")


def create_model(cfg, device: Optional[torch.device] = None, inference: bool = False):
    """Create model from config.

    Args:
        cfg: Config object (OmegaConf or dict) with model section.
        device: Optional device to move model to.
        inference: If True, sets class_dropout_prob=0 for inference.

    Returns:
        Model instance.
    """
    model_type = _get(cfg, "model", "type", default="mlp")

    if model_type == "mlp":
        from models import MLPDenoiser

        class_dropout = 0.0 if inference else _get(cfg, "model", "class_dropout_prob", default=0.1)
        model = MLPDenoiser(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth"),
            num_classes=_get(cfg, "model", "num_classes", default=0),
            class_dropout_prob=class_dropout,
        )

    elif model_type == "vae_mlp":
        from paradigms.variational_autoencoder import VAEModel

        model = VAEModel(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth", default=3),
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
        )

    elif model_type == "generator_mlp":
        from paradigms.generative_adversarial_network import Generator

        model = Generator(
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth", default=4),
            output_dim=_get(cfg, "model", "input_dim", default=2),
        )

    elif model_type == "ae_mlp":
        from paradigms.autoencoder import AEModel

        model = AEModel(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth", default=3),
            latent_dim=_get(cfg, "model", "latent_dim", default=16),
        )

    elif model_type == "shortcut_mlp":
        from models import ShortcutMLPDenoiser

        class_dropout = 0.0 if inference else _get(cfg, "model", "class_dropout_prob", default=0.1)
        model = ShortcutMLPDenoiser(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth"),
            num_classes=_get(cfg, "model", "num_classes", default=0),
            class_dropout_prob=class_dropout,
        )

    elif model_type == "energy_mlp":
        from paradigms.energy_based_model import EnergyMLP

        model = EnergyMLP(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size", default=256),
            depth=_get(cfg, "model", "depth", default=6),
            use_spectral_norm=_get(cfg, "model", "use_spectral_norm", default=True),
        )

    elif model_type == "ar_transformer":
        from paradigms.autoregressive import ARTransformer2D

        class_dropout = 0.0 if inference else _get(cfg, "model", "class_dropout_prob", default=0.0)
        model = ARTransformer2D(
            num_bins=_get(cfg, "model", "num_bins", default=128),
            hidden_size=_get(cfg, "model", "hidden_size"),
            depth=_get(cfg, "model", "depth", default=6),
            num_heads=_get(cfg, "model", "num_heads", default=4),
            input_dim=_get(cfg, "model", "input_dim", default=2),
            num_classes=_get(cfg, "model", "num_classes", default=0),
            class_dropout_prob=class_dropout,
        )

    elif model_type == "drifting_mlp":
        from paradigms.drifting_model import DriftingGenerator

        model = DriftingGenerator(
            latent_dim=_get(cfg, "model", "latent_dim", default=32),
            hidden_size=_get(cfg, "model", "hidden_size", default=256),
            depth=_get(cfg, "model", "depth", default=3),
            output_dim=_get(cfg, "model", "input_dim", default=2),
        )

    elif model_type == "meanflow_mlp":
        from models import MeanFlowMLPDenoiser

        model = MeanFlowMLPDenoiser(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size", default=256),
            depth=_get(cfg, "model", "depth", default=6),
        )

    elif model_type == "imf_mlp":
        from models import iMFMLPDenoiser

        model = iMFMLPDenoiser(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size", default=256),
            shared_depth=_get(cfg, "model", "shared_depth", default=2),
            head_depth=_get(cfg, "model", "head_depth", default=4),
        )

    elif model_type == "realnvp":
        from paradigms.normalizing_flow import RealNVP2D

        model = RealNVP2D(
            input_dim=_get(cfg, "model", "input_dim", default=2),
            hidden_size=_get(cfg, "model", "hidden_size", default=128),
            num_coupling_layers=_get(cfg, "model", "num_coupling_layers", default=8),
            coupling_depth=_get(cfg, "model", "coupling_depth", default=2),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if device is not None:
        model = model.to(device)

    return model


def create_sampler(
    sampler_type: Union[str, SamplerType],
    schedule,
    parameterization,
    eta: float = 0.0,
):
    """Create sampler for diffusion models.

    Args:
        sampler_type: Type of sampler ("ddpm" or "ddim").
        schedule: Noise schedule from DDPM.
        parameterization: What the model predicts (eps, x0, v).
        eta: DDIM stochasticity (0=deterministic, 1=DDPM-like). Only used for DDIM.

    Returns:
        DDPMSampler or DDIMSampler instance.
    """
    from paradigms.diffusion import DDPMSampler, DDIMSampler

    sampler_type = SamplerType(sampler_type)

    if sampler_type == SamplerType.DDPM:
        return DDPMSampler(schedule, parameterization)

    elif sampler_type == SamplerType.DDIM:
        return DDIMSampler(schedule, parameterization, eta=eta)

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_optimizer(model, cfg):
    """Create optimizer from config.

    Args:
        model: Model to optimize.
        cfg: Config object (OmegaConf or dict) with training.optimizer section.

    Returns:
        Optimizer instance.

    Supported optimizers:
        - adamw: AdamW optimizer
    """
    opt_type = _get(cfg, "training", "optimizer", "type")

    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=_get(cfg, "training", "optimizer", "lr"),
            betas=(
                _get(cfg, "training", "optimizer", "beta1", default=0.9),
                _get(cfg, "training", "optimizer", "beta2", default=0.999),
            ),
            weight_decay=_get(cfg, "training", "optimizer", "weight_decay", default=0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
