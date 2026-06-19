#!/usr/bin/env bash
# Run PaddleOCR-VL on H100 for an existing v2 benchmark run, then re-score.
#
# Usage (on H100):
#   export CUDA_VISIBLE_DEVICES=6
#   bash ocr_benchmark_v2/scripts/run_h100_paddle.sh 20260618T103056Z
set -euo pipefail

RUN_ID="${1:?run_id required, e.g. 20260618T103056Z}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/h100_env.sh"
export OCR_HF_LOCAL_ONLY=0

FREE_GPU="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$FREE_GPU" ]]; then
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '{print $1,$2}' | sort -k2 -n | head -1 | awk '{print $1}')
  export CUDA_VISIBLE_DEVICES="$FREE_GPU"
fi

echo "=== GPU $CUDA_VISIBLE_DEVICES ==="
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,memory.total --format=csv

cd "$REPO_ROOT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Plan paddleocr-vl ==="
python -m ocr_benchmark_v2 run plan --run-id "$RUN_ID" --model paddleocr-vl

echo "=== PaddleOCR-VL ==="
python -m ocr_benchmark_v2 run execute --run-id "$RUN_ID" --model paddleocr-vl

echo "=== Re-score + report ==="
python -m ocr_benchmark_v2 run score --run-id "$RUN_ID"
python -m ocr_benchmark_v2 run report --run-id "$RUN_ID"

echo "=== Done ==="
ls -la "$REPO_ROOT/runs/$RUN_ID/outputs/paddleocr-vl/"
ls -la "$REPO_ROOT/runs/$RUN_ID/scores/"
