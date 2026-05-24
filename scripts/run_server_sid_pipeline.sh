#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Automotive}"
K_CORE="${K_CORE:-5}"
DATASET_NAME="${DATASET_NAME:-Amazon_${CATEGORY}_priced_${K_CORE}core}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/home/youwen/data/rec/amazon_price_aware/processed_price_aware}"
PROCESSED_DIR="${PROCESSED_DIR:-${PROCESSED_ROOT}/${DATASET_NAME}}"
INDEX_DIR="${INDEX_DIR:-${PROCESSED_DIR}/index}"
MODEL_PATH="${MODEL_PATH:-/home/youwen/data/minionerec/models/Qwen2.5-1.5B}"
MINIONEREC_RQ_DIR="${MINIONEREC_RQ_DIR:-/home/youwen/work/Rec/MiniOneRec-exp/mini_one_rec_baseline/rq}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-512}"
RQ_EPOCHS="${RQ_EPOCHS:-500}"
RQ_BATCH_SIZE="${RQ_BATCH_SIZE:-2048}"
RQ_EVAL_BATCH_SIZE="${RQ_EVAL_BATCH_SIZE:-64}"

ITEM_JSON="${INDEX_DIR}/${DATASET_NAME}.item.json"
EMB_NPY="${INDEX_DIR}/${DATASET_NAME}.emb-qwen2p5-1p5b-td.npy"
EMB_IDS="${INDEX_DIR}/${DATASET_NAME}.emb-qwen2p5-1p5b-td.ids.json"
SID_INDEX="${INDEX_DIR}/${DATASET_NAME}.index.json"
RQ_CKPT_DIR="${PROCESSED_DIR}/rqvae_ckpt"

echo "dataset: ${DATASET_NAME}"
echo "processed_dir: ${PROCESSED_DIR}"

python scripts/generate_amazon_embeddings.py \
  --item_json "${ITEM_JSON}" \
  --output_npy "${EMB_NPY}" \
  --output_ids_json "${EMB_IDS}" \
  --model_path "${MODEL_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${EMBED_MAX_LENGTH}"

python scripts/generate_amazon_sid_index.py \
  --embedding_npy "${EMB_NPY}" \
  --ids_json "${EMB_IDS}" \
  --output_index_json "${SID_INDEX}" \
  --minionerec_rq_dir "${MINIONEREC_RQ_DIR}" \
  --ckpt_dir "${RQ_CKPT_DIR}" \
  --device "${DEVICE}" \
  --epochs "${RQ_EPOCHS}" \
  --batch_size "${RQ_BATCH_SIZE}" \
  --eval_batch_size "${RQ_EVAL_BATCH_SIZE}"

python scripts/export_amazon_minionerec_with_sid.py \
  --processed-dir "${PROCESSED_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --sid-index-path "${SID_INDEX}"
