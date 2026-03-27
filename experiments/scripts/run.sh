#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


LR_CONF=(
    0.02
    0.01
    0.002
)

GRAD_NOISE_CONF=(
    "BER3"
    "BER1"
    "NORM3"
    "NORM1"
    "UNIFORM"
    "ROUNDING"
)

CONFIGS=(
    "experiments/rfdn_sr/config.yaml"
    "experiments/resnet20_cifar10/config.yaml"
    "experiments/resnet20_cifar100/config.yaml"
)

RUNS_PER_PAIR=5


for cfg in "${CONFIGS[@]}"; do
    model_dir="$(dirname "${cfg}")"
    logs_folder="${model_dir}/logs"

    echo "============================================================"
    echo "Config : ${cfg}"
    echo "Logs   : ${logs_folder}"
    echo "============================================================"

    for grad_noise in "${GRAD_NOISE_CONF[@]}"; do
        for lr in "${LR_CONF[@]}"; do
            for run_idx in $(seq 1 "${RUNS_PER_PAIR}"); do
                echo "  [run ${run_idx}/${RUNS_PER_PAIR}] grad_noise=${grad_noise}  lr=${lr}"
                python -m scripts.gdnsq_q_config_v2 \
                    --config        "${cfg}" \
                    --grad-noise    "${grad_noise}" \
                    --lr            "${lr}" \
                    --mitrics-folder "${logs_folder}"
            done
        done
    done

    echo "Done: ${cfg}"
done

echo "All runs finished."
