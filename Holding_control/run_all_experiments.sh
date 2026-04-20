#!/bin/bash
# =============================================================================
# Unified experiment launcher: 4 methods × 3 seeds × 500 episodes
# Usage:
#   bash run_all_experiments.sh          # run all
#   bash run_all_experiments.sh sac      # run only SAC
#   bash run_all_experiments.sh lagrange # run only SAC-Lagrange
#   bash run_all_experiments.sh ensemble # run only Ensemble-SAC-Lagrange
#   bash run_all_experiments.sh noaction # run only No-Action baseline
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS=(0 1 2)
SIGMA=1.5
MAX_EPISODES=500
METHOD="${1:-all}"

RESULTS_DIR="paper_results"

run_sac() {
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "[SAC] seed=$seed starting..."
        echo "=========================================="
        python sac_v2_bus.py \
            --save_root "${RESULTS_DIR}/sac" \
            --run_name "paper" \
            --route_sigma $SIGMA \
            --max_episodes $MAX_EPISODES \
            --seed $seed \
            --hidden_dim 32 \
            --lr 1e-5 \
            --batch_size 2048 \
            --training_freq 10 \
            --plot_freq 5
        echo "[SAC] seed=$seed done."
    done
}

run_lagrange() {
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "[SAC-Lagrange] seed=$seed starting..."
        echo "=========================================="
        python sac_lagrange_bus.py \
            --save_root "${RESULTS_DIR}/sac_lagrange" \
            --run_name "paper" \
            --route_sigma $SIGMA \
            --max_episodes $MAX_EPISODES \
            --seed $seed \
            --hidden_dim 32 \
            --lr 1e-5 \
            --batch_size 2048 \
            --training_freq 10 \
            --plot_freq 5 \
            --cost_limit 20.0
        echo "[SAC-Lagrange] seed=$seed done."
    done
}

run_ensemble() {
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "[Ensemble-SAC-Lagrange] seed=$seed starting..."
        echo "=========================================="
        python sac_ensemble_lagrange.py \
            --save_root "${RESULTS_DIR}/ensemble" \
            --run_name "paper" \
            --route_sigma $SIGMA \
            --max_episodes $MAX_EPISODES \
            --seed $seed \
            --hidden_dim 64 \
            --ensemble_size 10 \
            --batch_size 2048 \
            --training_freq 5 \
            --plot_freq 1 \
            --cost_limit 20.0
        echo "[Ensemble-SAC-Lagrange] seed=$seed done."
    done
}

run_noaction() {
    for seed in "${SEEDS[@]}"; do
        echo "=========================================="
        echo "[No-Action] seed=$seed starting..."
        echo "=========================================="
        python no_action_baseline.py \
            --save_root "${RESULTS_DIR}/no_action" \
            --route_sigma $SIGMA \
            --num_episodes 30 \
            --seed $seed
        echo "[No-Action] seed=$seed done."
    done
}

case "$METHOD" in
    sac)       run_sac ;;
    lagrange)  run_lagrange ;;
    ensemble)  run_ensemble ;;
    noaction)  run_noaction ;;
    all)
        run_noaction
        run_sac
        run_lagrange
        run_ensemble
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: $0 {all|sac|lagrange|ensemble|noaction}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "=========================================="
