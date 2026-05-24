#!/usr/bin/env bash
set -euo pipefail

# Strictly serial nightly queue for mini5000 auxiliary-task ablation.
# Run from the server with:
#   setsid bash scripts/run_server_nightly_mini5000_aux_ablation.sh > logs/nightly_mini5000_aux_ablation.log 2>&1 < /dev/null &

REPO_DIR="${REPO_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-data/amazon_price_aware}"
BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-1.5B}"
DATASET="Amazon_Industrial_and_Scientific_priced_5core_mini5000"
PROCESSED_DIR="${DATA_ROOT}/processed_price_aware/${DATASET}"
TRAIN_FILE="${PROCESSED_DIR}/minionerec/train/${DATASET}.csv"
VALID_FILE="${PROCESSED_DIR}/minionerec/valid/${DATASET}.csv"
TEST_FILE="${PROCESSED_DIR}/minionerec/test/${DATASET}.csv"
ITEM_META="${PROCESSED_DIR}/index/${DATASET}.item.json"
SID_INDEX="${PROCESSED_DIR}/index/${DATASET}.index.json"
INFO_FILE="${PROCESSED_DIR}/info/${DATASET}.txt"
OUTPUT_ROOT="${DATA_ROOT}/outputs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
QUEUE_NAME="mini5000_aux_ablation_${RUN_ID}"
QUEUE_LOG_DIR="${DATA_ROOT}/logs/nightly_runs/${QUEUE_NAME}"
EVAL_ROOT="${OUTPUT_ROOT}/eval/${QUEUE_NAME}"

mkdir -p "${QUEUE_LOG_DIR}" "${EVAL_ROOT}"
cd "${REPO_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log "Missing required file: $1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    log "Missing required directory: $1"
    exit 1
  fi
}

check_inputs() {
  require_dir "${REPO_DIR}"
  require_dir "${BASE_MODEL}"
  require_file "${TRAIN_FILE}"
  require_file "${VALID_FILE}"
  require_file "${TEST_FILE}"
  require_file "${ITEM_META}"
  require_file "${SID_INDEX}"
  require_file "${INFO_FILE}"
}

train_baseline() {
  local exp_name="$1"
  local metadata_sample="$2"
  local fusion_sample="$3"
  local output_dir="${OUTPUT_ROOT}/baseline_sft/${DATASET}/${exp_name}_${RUN_ID}"
  local cache_dir="${OUTPUT_ROOT}/hf_cache/${DATASET}/${exp_name}_${RUN_ID}"
  local torchrun_log_dir="${QUEUE_LOG_DIR}/torchrun_${exp_name}"
  local train_log="${QUEUE_LOG_DIR}/${exp_name}.train.log"

  mkdir -p "${torchrun_log_dir}"
  log "TRAIN START ${exp_name}"
  log "train_log=${train_log}"

  CUDA_VISIBLE_DEVICES=0,3 torchrun \
    --nproc_per_node=2 \
    --log-dir "${torchrun_log_dir}" \
    --redirects 3 \
    scripts/train_amazon_baseline_sft.py \
    --base-model "${BASE_MODEL}" \
    --train-file "${TRAIN_FILE}" \
    --eval-file "${VALID_FILE}" \
    --item-meta-path "${ITEM_META}" \
    --sid-index-path "${SID_INDEX}" \
    --dataset-name "${DATASET}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-dir "${output_dir}" \
    --run-tag "${exp_name}" \
    --cache-dir "${cache_dir}" \
    --train-sample -1 \
    --metadata-sample "${metadata_sample}" \
    --fusion-sample "${fusion_sample}" \
    --eval-sample 2000 \
    --num-epochs 1 \
    --batch-size 32 \
    --micro-batch-size 1 \
    --cutoff-len 512 \
    --learning-rate 1e-5 \
    --eval-steps 0.25 \
    --save-steps 0.25 \
    --gradient-checkpointing \
    > "${train_log}" 2>&1

  local checkpoint="${output_dir}/final_checkpoint"
  if [[ ! -d "${checkpoint}" ]]; then
    log "TRAIN FAILED: missing final checkpoint for ${exp_name}: ${checkpoint}"
    exit 1
  fi

  log "TRAIN DONE ${exp_name}"
  log "checkpoint=${checkpoint}"
  echo "${checkpoint}"
}

eval_checkpoint() {
  local exp_name="$1"
  local checkpoint="$2"
  local eval_dir="${EVAL_ROOT}/${exp_name}"
  local eval_log="${QUEUE_LOG_DIR}/${exp_name}.eval.log"
  local predictions="${eval_dir}/predictions.json"
  local metrics="${eval_dir}/metrics.json"

  mkdir -p "${eval_dir}"
  log "EVAL START ${exp_name}"
  log "eval_log=${eval_log}"

  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_amazon_sft.py \
    --base-model "${checkpoint}" \
    --info-file "${INFO_FILE}" \
    --test-data-path "${TEST_FILE}" \
    --item-meta-path "${ITEM_META}" \
    --result-json-data "${predictions}" \
    --metrics-json-data "${metrics}" \
    --sample -1 \
    --batch-size 8 \
    --num-beams 20 \
    --max-new-tokens 64 \
    --length-penalty 0.0 \
    > "${eval_log}" 2>&1

  require_file "${metrics}"
  log "EVAL DONE ${exp_name}"
  log "metrics=${metrics}"
  {
    echo "=== ${exp_name} ==="
    echo "checkpoint=${checkpoint}"
    echo "metrics=${metrics}"
    cat "${metrics}"
    echo
  } >> "${QUEUE_LOG_DIR}/summary.txt"
}

run_experiment() {
  local exp_name="$1"
  local metadata_sample="$2"
  local fusion_sample="$3"
  local checkpoint

  checkpoint="$(train_baseline "${exp_name}" "${metadata_sample}" "${fusion_sample}")"
  eval_checkpoint "${exp_name}" "${checkpoint}"
}

log "QUEUE START ${QUEUE_NAME}"
log "repo=${REPO_DIR}"
log "summary=${QUEUE_LOG_DIR}/summary.txt"
check_inputs

run_experiment "baseline_1p5b_full_sft_mini5000_aux3000_3000" 3000 3000
run_experiment "baseline_1p5b_full_sft_mini5000_seqonly" 0 0

log "QUEUE DONE ${QUEUE_NAME}"
log "summary=${QUEUE_LOG_DIR}/summary.txt"
