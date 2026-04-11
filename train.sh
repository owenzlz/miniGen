#!/bin/bash

# 2D Synthetic Experiments (SwissRoll)
python train.py --config configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/diffusion/DDIMSampler_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/variational_autoencoder/VAE_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/generative_adversarial_network/GAN_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/score_matching_ncsn/ScoreMatchingNCSN_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/score_matching_hyvarinen/ScoreMatchingHyvarinen_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/normalizing_flow/NormalizingFlow_RealNVP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/continuous_normalizing_flow/CNF_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/consistency_training/ConsistencyTraining_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/autoencoder/AE_MLP_SwissRoll.yaml --skip_save_ckpt
python train.py --config configs/improved_meanflow/iMF_MLP_SwissRoll.yaml --skip_save_ckpt
