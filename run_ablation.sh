#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:?Usage: bash scripts/run_ablation.sh data/dataset.csv}"

for VARIANT in full no_role no_interaction no_adaptive no_role_interaction; do
  python -m src.train \
    --data "$DATA_PATH" \
    --output_dir "outputs/ablation_${VARIANT}" \
    --variant "$VARIANT" \
    --model_name bert-base-uncased \
    --max_length 128 \
    --batch_size 8 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --seed 42
done

