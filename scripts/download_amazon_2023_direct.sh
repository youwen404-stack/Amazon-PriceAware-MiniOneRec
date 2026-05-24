#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Office_Products}"
OUTPUT_DIR="${OUTPUT_DIR:-data/raw}"
HF_ROOT="${HF_ROOT:-https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main}"

REVIEW_URL="${HF_ROOT}/raw/review_categories/${CATEGORY}.jsonl"
META_URL="${HF_ROOT}/raw/meta_categories/meta_${CATEGORY}.jsonl"

mkdir -p "${OUTPUT_DIR}"

echo "[download] category: ${CATEGORY}"
echo "[download] output: ${OUTPUT_DIR}"
echo "[download] review url: ${REVIEW_URL}"
echo "[download] meta url: ${META_URL}"

curl -L --fail --retry 50 --retry-delay 10 --retry-all-errors --connect-timeout 30 -C - \
  "${REVIEW_URL}" \
  -o "${OUTPUT_DIR}/${CATEGORY}.reviews.jsonl"

curl -L --fail --retry 50 --retry-delay 10 --retry-all-errors --connect-timeout 30 -C - \
  "${META_URL}" \
  -o "${OUTPUT_DIR}/meta_${CATEGORY}.jsonl"

echo "[download] files:"
ls -lh "${OUTPUT_DIR}/${CATEGORY}.reviews.jsonl" "${OUTPUT_DIR}/meta_${CATEGORY}.jsonl"
