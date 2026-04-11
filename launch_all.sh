#!/bin/bash
# Launch all experiments across 8 GPUs (max 2 concurrent per GPU)
# Within each paradigm, datasets run sequentially (Checkerboard → GaussianMixture → SwissRoll)

set -e
cd /sensei-fs-3/users/lingzzha/ML/Generative_Playground_2D

TRAIN="python train.py --skip_save_ckpt --config"

# Helper: run a paradigm's configs sequentially on a specific GPU
run_paradigm() {
    local gpu=$1
    shift
    for cfg in "$@"; do
        echo "[GPU $gpu] Starting: $cfg"
        CUDA_VISIBLE_DEVICES=$gpu $TRAIN "$cfg"
        echo "[GPU $gpu] Finished: $cfg"
    done
}

# ── GPU 0: alpha_flow → shortcut_model  ||  autoencoder ──
run_paradigm 0 \
    configs/alpha_flow/AlphaFlow_MLP_Checkerboard.yaml \
    configs/alpha_flow/AlphaFlow_MLP_GaussianMixture.yaml \
    configs/alpha_flow/AlphaFlow_MLP_SwissRoll.yaml \
    configs/shortcut_model/ShortcutModel_MLP_Checkerboard.yaml \
    configs/shortcut_model/ShortcutModel_MLP_GaussianMixture.yaml \
    configs/shortcut_model/ShortcutModel_MLP_SwissRoll.yaml &

run_paradigm 0 \
    configs/autoencoder/AE_MLP_Checkerboard.yaml \
    configs/autoencoder/AE_MLP_GaussianMixture.yaml \
    configs/autoencoder/AE_MLP_SwissRoll.yaml &

# ── GPU 1: autoregressive → variational_autoencoder  ||  consistency_training ──
run_paradigm 1 \
    configs/autoregressive/Autoregressive_Transformer_Checkerboard.yaml \
    configs/autoregressive/Autoregressive_Transformer_GaussianMixture.yaml \
    configs/autoregressive/Autoregressive_Transformer_SwissRoll.yaml \
    configs/variational_autoencoder/VAE_MLP_Checkerboard.yaml \
    configs/variational_autoencoder/VAE_MLP_GaussianMixture.yaml \
    configs/variational_autoencoder/VAE_MLP_SwissRoll.yaml &

run_paradigm 1 \
    configs/consistency_training/ConsistencyTraining_MLP_Checkerboard.yaml \
    configs/consistency_training/ConsistencyTraining_MLP_GaussianMixture.yaml \
    configs/consistency_training/ConsistencyTraining_MLP_SwissRoll.yaml &

# ── GPU 2: continuous_normalizing_flow  ||  drifting_model ──
run_paradigm 2 \
    configs/continuous_normalizing_flow/CNF_MLP_Checkerboard.yaml \
    configs/continuous_normalizing_flow/CNF_MLP_GaussianMixture.yaml \
    configs/continuous_normalizing_flow/CNF_MLP_SwissRoll.yaml &

run_paradigm 2 \
    configs/drifting_model/DriftingModel_MLP_Checkerboard.yaml \
    configs/drifting_model/DriftingModel_MLP_GaussianMixture.yaml \
    configs/drifting_model/DriftingModel_MLP_SwissRoll.yaml &

# ── GPU 3: diffusion (DDPM + DDIM, 6 configs)  ||  energy_based_model ──
run_paradigm 3 \
    configs/diffusion/DDPMSampler_MLP_Checkerboard.yaml \
    configs/diffusion/DDPMSampler_MLP_GaussianMixture.yaml \
    configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml \
    configs/diffusion/DDIMSampler_MLP_Checkerboard.yaml \
    configs/diffusion/DDIMSampler_MLP_GaussianMixture.yaml \
    configs/diffusion/DDIMSampler_MLP_SwissRoll.yaml &

run_paradigm 3 \
    configs/energy_based_model/EBM_MLP_Checkerboard.yaml \
    configs/energy_based_model/EBM_MLP_GaussianMixture.yaml \
    configs/energy_based_model/EBM_MLP_SwissRoll.yaml &

# ── GPU 4: equilibrium_matching  ||  flow_matching ──
run_paradigm 4 \
    configs/equilibrium_matching/EqM_MLP_Checkerboard.yaml \
    configs/equilibrium_matching/EqM_MLP_GaussianMixture.yaml \
    configs/equilibrium_matching/EqM_MLP_SwissRoll.yaml &

run_paradigm 4 \
    configs/flow_matching/FlowMatching_MLP_Checkerboard.yaml \
    configs/flow_matching/FlowMatching_MLP_GaussianMixture.yaml \
    configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml &

# ── GPU 5: GAN  ||  improved_meanflow ──
run_paradigm 5 \
    configs/generative_adversarial_network/GAN_MLP_Checkerboard.yaml \
    configs/generative_adversarial_network/GAN_MLP_GaussianMixture.yaml \
    configs/generative_adversarial_network/GAN_MLP_SwissRoll.yaml &

run_paradigm 5 \
    configs/improved_meanflow/iMF_MLP_Checkerboard.yaml \
    configs/improved_meanflow/iMF_MLP_GaussianMixture.yaml \
    configs/improved_meanflow/iMF_MLP_SwissRoll.yaml &

# ── GPU 6: meanflow  ||  normalizing_flow ──
run_paradigm 6 \
    configs/meanflow/MeanFlow_MLP_Checkerboard.yaml \
    configs/meanflow/MeanFlow_MLP_GaussianMixture.yaml \
    configs/meanflow/MeanFlow_MLP_SwissRoll.yaml &

run_paradigm 6 \
    configs/normalizing_flow/NormalizingFlow_RealNVP_Checkerboard.yaml \
    configs/normalizing_flow/NormalizingFlow_RealNVP_GaussianMixture.yaml \
    configs/normalizing_flow/NormalizingFlow_RealNVP_SwissRoll.yaml &

# ── GPU 7: score_matching_hyvarinen  ||  score_matching_ncsn ──
run_paradigm 7 \
    configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_Checkerboard.yaml \
    configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_GaussianMixture.yaml \
    configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_SwissRoll.yaml &

run_paradigm 7 \
    configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_Checkerboard.yaml \
    configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_GaussianMixture.yaml \
    configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_SwissRoll.yaml &

echo "All 57 experiments launched across 8 GPUs (18 paradigms, 3 datasets each + diffusion 6)"
echo "Waiting for all to complete..."
wait
echo "All experiments finished!"
