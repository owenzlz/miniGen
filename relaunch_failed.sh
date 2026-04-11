#!/bin/bash
# Re-launch experiments that failed due to gaussian_mixture num_classes=0 bug (now fixed)
# 11 paradigm chains died on GaussianMixture → need to re-run GM + SwissRoll for those
# Plus variational_autoencoder (all 3) and DDIM (all 3) that never started

set -e
cd /sensei-fs-3/users/lingzzha/ML/Generative_Playground_2D

TRAIN="python train.py --skip_save_ckpt --config"

run_paradigm() {
    local gpu=$1
    shift
    for cfg in "$@"; do
        echo "[GPU $gpu] Starting: $cfg"
        CUDA_VISIBLE_DEVICES=$gpu $TRAIN "$cfg"
        echo "[GPU $gpu] Finished: $cfg"
    done
}

# ── GPU 0 (1 slot free): autoencoder GM + SR ──
run_paradigm 0 \
    configs/autoencoder/AE_MLP_GaussianMixture.yaml \
    configs/autoencoder/AE_MLP_SwissRoll.yaml &

# ── GPU 1 (2 slots free): autoregressive GM+SR → VAE all 3  ||  consistency_training GM+SR ──
run_paradigm 1 \
    configs/autoregressive/Autoregressive_Transformer_GaussianMixture.yaml \
    configs/autoregressive/Autoregressive_Transformer_SwissRoll.yaml \
    configs/variational_autoencoder/VAE_MLP_Checkerboard.yaml \
    configs/variational_autoencoder/VAE_MLP_GaussianMixture.yaml \
    configs/variational_autoencoder/VAE_MLP_SwissRoll.yaml &

run_paradigm 1 \
    configs/consistency_training/ConsistencyTraining_MLP_GaussianMixture.yaml \
    configs/consistency_training/ConsistencyTraining_MLP_SwissRoll.yaml &

# ── GPU 2 (1 slot free): drifting_model GM + SR ──
run_paradigm 2 \
    configs/drifting_model/DriftingModel_MLP_GaussianMixture.yaml \
    configs/drifting_model/DriftingModel_MLP_SwissRoll.yaml &

# ── GPU 3 (1 slot free): DDPM GM+SR → DDIM all 3 ──
run_paradigm 3 \
    configs/diffusion/DDPMSampler_MLP_GaussianMixture.yaml \
    configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml \
    configs/diffusion/DDIMSampler_MLP_Checkerboard.yaml \
    configs/diffusion/DDIMSampler_MLP_GaussianMixture.yaml \
    configs/diffusion/DDIMSampler_MLP_SwissRoll.yaml &

# ── GPU 4 (2 slots free): equilibrium_matching GM+SR  ||  flow_matching GM+SR ──
run_paradigm 4 \
    configs/equilibrium_matching/EqM_MLP_GaussianMixture.yaml \
    configs/equilibrium_matching/EqM_MLP_SwissRoll.yaml &

run_paradigm 4 \
    configs/flow_matching/FlowMatching_MLP_GaussianMixture.yaml \
    configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml &

# ── GPU 5 (1 slot free): GAN GM + SR ──
run_paradigm 5 \
    configs/generative_adversarial_network/GAN_MLP_GaussianMixture.yaml \
    configs/generative_adversarial_network/GAN_MLP_SwissRoll.yaml &

# ── GPU 6 (1 slot free): normalizing_flow GM + SR ──
run_paradigm 6 \
    configs/normalizing_flow/NormalizingFlow_RealNVP_GaussianMixture.yaml \
    configs/normalizing_flow/NormalizingFlow_RealNVP_SwissRoll.yaml &

# ── GPU 7 (2 slots free): score_matching_hyvarinen GM+SR  ||  score_matching_ncsn GM+SR ──
run_paradigm 7 \
    configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_GaussianMixture.yaml \
    configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_SwissRoll.yaml &

run_paradigm 7 \
    configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_GaussianMixture.yaml \
    configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_SwissRoll.yaml &

echo "Re-launched 28 experiments (11 failed GM + 11 skipped SR + 6 never-started) across 8 GPUs"
echo "Waiting for all to complete..."
wait
echo "All re-launched experiments finished!"
