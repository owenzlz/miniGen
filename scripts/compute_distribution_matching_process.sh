#!/bin/bash
# =============================================================================
# Compute distribution matching process for drifting model on all 9 2D shapes.
# Runs all 9 distributions in parallel across GPUs 0-7.
#
# Usage:
#   bash scripts/compute_distribution_matching_process.sh
#
# Output: assets/drifting_field_distribution_matching/{dataset_name}/drifting_model/
# =============================================================================

SCRIPT="scripts/compute_distribution_matching_process.py"
COMMON_ARGS="--type drifting_model --total_steps 5000 --fps 3 --hold_last_s 1 --minimal --output_dir assets/drifting_field_distribution_matching"

run_shape() {
    local gpu=$1
    local shape=$2
    echo "[GPU $gpu] Starting: $shape"
    CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --dataset "$shape" $COMMON_ARGS
    echo "[GPU $gpu] Finished: $shape"
}

# 9 shapes across 8 GPUs (GPU 0 runs 2 sequentially)
run_shape 0 swiss_roll && run_shape 0 gaussian_mixture &
run_shape 1 moons &
run_shape 2 circles &
run_shape 3 checkerboard &
run_shape 4 spirals &
run_shape 5 pinwheel &
run_shape 6 rings &
run_shape 7 s_curve &

wait
echo "All 9 drifting model distribution matching processes complete."
