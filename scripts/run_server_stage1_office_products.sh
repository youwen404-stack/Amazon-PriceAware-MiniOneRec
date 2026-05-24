#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-data/amazon_price_aware}"
CATEGORY="${CATEGORY:-Office_Products}"

RAW_DIR="${DATA_DIR}/raw"
LOG_DIR="${DATA_DIR}/logs"
USE_PROJECT_VENV="${USE_PROJECT_VENV:-0}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${RAW_DIR}" "${LOG_DIR}"
cd "${PROJECT_DIR}"

if [[ "${USE_PROJECT_VENV}" == "1" ]]; then
  PYTHON="${PROJECT_DIR}/.venv/bin/python"
fi

if [[ "${USE_PROJECT_VENV}" == "1" && ! -x "${PYTHON}" ]]; then
  python3 -m venv .venv
  "${PYTHON}" -m pip install "datasets==2.19.2"
fi

echo "[stage1] project: ${PROJECT_DIR}"
echo "[stage1] data: ${DATA_DIR}"
echo "[stage1] category: ${CATEGORY}"
echo "[stage1] python: ${PYTHON}"

"${PYTHON}" scripts/download_amazon_2023.py \
  --category "${CATEGORY}" \
  --output_dir "${RAW_DIR}" \
  2>&1 | tee "${LOG_DIR}/download_${CATEGORY}.log"

"${PYTHON}" scripts/audit_amazon_raw.py \
  --reviews "${RAW_DIR}/${CATEGORY}.reviews.jsonl" \
  --meta "${RAW_DIR}/meta_${CATEGORY}.jsonl" \
  2>&1 | tee "${LOG_DIR}/audit_raw_${CATEGORY}.log"

echo "[stage1] raw files:"
ls -lh "${RAW_DIR}/${CATEGORY}.reviews.jsonl" "${RAW_DIR}/meta_${CATEGORY}.jsonl"
echo "[stage1] logs:"
ls -lh "${LOG_DIR}/download_${CATEGORY}.log" "${LOG_DIR}/audit_raw_${CATEGORY}.log"
