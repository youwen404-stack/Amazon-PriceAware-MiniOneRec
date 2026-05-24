#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
K_CORE="${K_CORE:-5}"
CANDIDATE_ITEMS="${CANDIDATE_ITEMS:-5000}"
MIN_HISTORY_LEN="${MIN_HISTORY_LEN:-3}"
MAX_HISTORY_LEN="${MAX_HISTORY_LEN:-50}"
MAX_TRAIN_TARGETS_PER_USER="${MAX_TRAIN_TARGETS_PER_USER:-50}"
DATA_ROOT="${DATA_ROOT:-/home/youwen/data/rec/amazon_price_aware/raw_amazon_2023}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/youwen/data/rec/amazon_price_aware/processed_price_aware}"
DATASET_NAME="${DATASET_NAME:-Amazon_${CATEGORY}_priced_${K_CORE}core_mini${CANDIDATE_ITEMS}}"

python scripts/convert_amazon_to_minionerec_mini.py \
  --category "$CATEGORY" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --k-core "$K_CORE" \
  --candidate-items "$CANDIDATE_ITEMS" \
  --min-history-len "$MIN_HISTORY_LEN" \
  --max-history-len "$MAX_HISTORY_LEN" \
  --max-train-targets-per-user "$MAX_TRAIN_TARGETS_PER_USER"
