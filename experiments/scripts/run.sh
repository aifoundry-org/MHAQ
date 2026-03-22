#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS=(
  "experiments/resnet20_cifar10/configs/conf_lr_0_002.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running: ${cfg}"
  echo "============================================================"

  cfg_name=$(basename "$cfg" .yaml)
  python -m scripts.gdnsq_q_config --config "$cfg" > "experiments/${cfg_name}.log" 2>&1

  echo "Done: ${cfg}"
done

echo "All runs finished."
