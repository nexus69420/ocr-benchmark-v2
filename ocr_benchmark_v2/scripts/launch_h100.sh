#!/usr/bin/env bash
set -euo pipefail
source /amulpfsdata/models/ocr-benchmark/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export OCR_HF_LOCAL_ONLY=1
cd /home/aicloud/amul-oan-api
exec bash ocr_benchmark_v2/scripts/run_h100_pending.sh "${1:-20260618T103056Z}"
