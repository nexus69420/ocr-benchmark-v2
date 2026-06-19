#!/usr/bin/env bash
# Run pending Chandra + Qwen on H100 for an existing v2 run.
#
# Usage (on H100):
#   export CUDA_VISIBLE_DEVICES=6   # pick a free GPU
#   bash ocr_benchmark_v2/scripts/run_h100_pending.sh 20260618T103056Z
#
set -euo pipefail

RUN_ID="${1:?run_id required, e.g. 20260618T103056Z}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V2_ROOT/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/h100_env.sh"

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

echo "=== Plan ==="
python -m ocr_benchmark_v2 run plan --run-id "$RUN_ID" --model chandra-2 --model qwen2.5-vl-7b

echo "=== Chandra OCR 2 ==="
python -m ocr_benchmark_v2 run execute --run-id "$RUN_ID" --model chandra-2

echo "=== Qwen2.5-VL-7B ==="
python -m ocr_benchmark_v2 run execute --run-id "$RUN_ID" --model qwen2.5-vl-7b

echo "=== Re-score + report ==="
python -m ocr_benchmark_v2 run score --run-id "$RUN_ID"
python -m ocr_benchmark_v2 run report --run-id "$RUN_ID"

echo "=== Done ==="
ls -la "$REPO_ROOT/runs/$RUN_ID/scores/"
