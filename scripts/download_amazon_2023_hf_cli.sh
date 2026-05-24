#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Office_Products}"
OUTPUT_DIR="${OUTPUT_DIR:-data/raw}"
LOCAL_DIR="${LOCAL_DIR:-data/hf_download}"
HF_BIN="${HF_BIN:-.venv/bin/hf}"
REPO_ID="${REPO_ID:-McAuley-Lab/Amazon-Reviews-2023}"

REVIEW_PATH="raw/review_categories/${CATEGORY}.jsonl"
META_PATH="raw/meta_categories/meta_${CATEGORY}.jsonl"

mkdir -p "${OUTPUT_DIR}" "${LOCAL_DIR}"

echo "[hf-download] category: ${CATEGORY}"
echo "[hf-download] local dir: ${LOCAL_DIR}"
echo "[hf-download] output dir: ${OUTPUT_DIR}"
echo "[hf-download] hf bin: ${HF_BIN}"

"${HF_BIN}" download "${REPO_ID}" \
  "${REVIEW_PATH}" \
  "${META_PATH}" \
  --repo-type dataset \
  --local-dir "${LOCAL_DIR}" \
  --max-workers 2

cp "${LOCAL_DIR}/${REVIEW_PATH}" "${OUTPUT_DIR}/${CATEGORY}.reviews.jsonl"
cp "${LOCAL_DIR}/${META_PATH}" "${OUTPUT_DIR}/meta_${CATEGORY}.jsonl"

echo "[hf-download] files:"
ls -lh "${OUTPUT_DIR}/${CATEGORY}.reviews.jsonl" "${OUTPUT_DIR}/meta_${CATEGORY}.jsonl"
