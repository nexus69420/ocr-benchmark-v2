#!/usr/bin/env bash
# Source on H100 before v2 GPU runs. Models live on extended disk:
#   /amulpfsdata/models/ocr-benchmark/hf-cache/hub/
set -euo pipefail

ENV_FILE="${OCR_BENCH_ENV:-/amulpfsdata/models/ocr-benchmark/env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "Missing $ENV_FILE — run migrate_to_amulpfsdata.sh first" >&2
  exit 1
fi

# Load weights from attached disk only (no Hub download).
export OCR_HF_LOCAL_ONLY="${OCR_HF_LOCAL_ONLY:-1}"

echo "OCR_BENCH_ROOT=$OCR_BENCH_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "OCR_HF_LOCAL_ONLY=$OCR_HF_LOCAL_ONLY"

export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$OCR_BENCH_ROOT/paddle-cache}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
echo "PADDLE_PDX_CACHE_HOME=$PADDLE_PDX_CACHE_HOME"

for name in \
  models--Qwen--Qwen2.5-VL-7B-Instruct \
  models--datalab-to--chandra-ocr-2; do
  if [[ ! -d "$HF_HUB_CACHE/$name" ]]; then
    echo "WARN: missing cache dir $HF_HUB_CACHE/$name" >&2
  else
    echo "ok $name"
  fi
done

if [[ -d "$PADDLE_PDX_CACHE_HOME" ]]; then
  echo "ok paddle-cache ($(du -sh "$PADDLE_PDX_CACHE_HOME" 2>/dev/null | awk '{print $1}'))"
else
  echo "paddle-cache: not yet populated (first PaddleOCR-VL run will download)"
fi
