#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Automotive}"
K_CORE="${K_CORE:-5}"
DATASET_NAME="${DATASET_NAME:-Amazon_${CATEGORY}_priced_${K_CORE}core}"
DATA_ROOT="${DATA_ROOT:-/home/youwen/data/rec/amazon_price_aware}"
PROCESSED_DIR="${PROCESSED_DIR:-${DATA_ROOT}/processed_price_aware/${DATASET_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs}"
BASE_MODEL="${BASE_MODEL:-/home/youwen/data/minionerec/models/Qwen2.5-1.5B}"

TRAIN_SAMPLE="${TRAIN_SAMPLE:-50000}"
METADATA_SAMPLE="${METADATA_SAMPLE:-${TRAIN_SAMPLE}}"
FUSION_SAMPLE="${FUSION_SAMPLE:-${TRAIN_SAMPLE}}"
EVAL_SAMPLE="${EVAL_SAMPLE:-5000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
SEED="${SEED:-42}"
FREEZE_LLM="${FREEZE_LLM:-1}"
if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "$FREEZE_LLM" == "1" || "$FREEZE_LLM" == "true" || "$FREEZE_LLM" == "TRUE" ]]; then
    RUN_TAG="smoke_freeze"
  else
    RUN_TAG="smoke_fullft"
  fi
fi

EXTRA_ARGS=()
if [[ "$FREEZE_LLM" == "1" || "$FREEZE_LLM" == "true" || "$FREEZE_LLM" == "TRUE" ]]; then
  EXTRA_ARGS+=(--freeze-LLM)
fi

python scripts/train_amazon_baseline_sft.py \
  --base-model "${BASE_MODEL}" \
  --train-file "${PROCESSED_DIR}/minionerec/train/${DATASET_NAME}.csv" \
  --eval-file "${PROCESSED_DIR}/minionerec/valid/${DATASET_NAME}.csv" \
  --item-meta-path "${PROCESSED_DIR}/index/${DATASET_NAME}.item.json" \
  --sid-index-path "${PROCESSED_DIR}/index/${DATASET_NAME}.index.json" \
  --dataset-name "${DATASET_NAME}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --train-sample "${TRAIN_SAMPLE}" \
  --metadata-sample "${METADATA_SAMPLE}" \
  --fusion-sample "${FUSION_SAMPLE}" \
  --eval-sample "${EVAL_SAMPLE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --cutoff-len "${CUTOFF_LEN}" \
  --learning-rate "${LEARNING_RATE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
