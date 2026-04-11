"""Shared utilities for visualization and computation scripts."""

# Map paradigm type to its default config file (relative to project root)
PARADIGM_TO_CONFIG = {
    "ddpm": "configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml",
    "flow_matching": "configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml",
    "variational_autoencoder": "configs/variational_autoencoder/VAE_MLP_SwissRoll.yaml",
    "generative_adversarial_network": "configs/generative_adversarial_network/GAN_MLP_SwissRoll.yaml",
    "score_matching_ncsn": "configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_SwissRoll.yaml",
    "score_matching_hyvarinen": "configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_SwissRoll.yaml",
    "normalizing_flow": "configs/normalizing_flow/NormalizingFlow_RealNVP_SwissRoll.yaml",
    "continuous_normalizing_flow": "configs/continuous_normalizing_flow/CNF_MLP_SwissRoll.yaml",
    "consistency_training": "configs/consistency_training/ConsistencyTraining_MLP_SwissRoll.yaml",
    "autoencoder": "configs/autoencoder/AE_MLP_SwissRoll.yaml",
    "shortcut_model": "configs/shortcut_model/ShortcutModel_MLP_SwissRoll.yaml",
    "drifting_model": "configs/drifting_model/DriftingModel_MLP_SwissRoll.yaml",
    "energy_based_model": "configs/energy_based_model/EBM_MLP_SwissRoll.yaml",
    "improved_meanflow": "configs/improved_meanflow/iMF_MLP_SwissRoll.yaml",
    "meanflow": "configs/meanflow/MeanFlow_MLP_SwissRoll.yaml",
    "alpha_flow": "configs/alpha_flow/AlphaFlow_MLP_SwissRoll.yaml",
    "autoregressive": "configs/autoregressive/Autoregressive_Transformer_SwissRoll.yaml",
}


def compute_save_steps(total_steps, num_frames):
    """Compute which training steps to save, with frequency inversely proportional to time.

    Uses a quadratic schedule: step_k = total_steps * (k / (N-1))^2.
    This bunches frames at the start (frequent early, sparse later).

    Returns:
        Set of training steps at which to save a frame.
    """
    if num_frames <= 1:
        return {total_steps - 1}
    save_steps = set()
    for k in range(num_frames):
        step = int(total_steps * (k / (num_frames - 1)) ** 2)
        step = min(step, total_steps - 1)
        save_steps.add(step)
    return save_steps
