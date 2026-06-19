#!/usr/bin/env bash
set -euo pipefail
source ~/ocr-benchmark/env.sh
echo "=== python ==="
python --version
echo "=== torch ==="
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
pip list | grep -iE 'torch|transformers|paddle' || true
echo "=== cache ==="
ls ~/ocr-benchmark/hf-cache/hub/ 2>/dev/null | head -10
ls ~/ocr-benchmark/paddle-cache 2>/dev/null || echo "no paddle-cache yet"
df -h ~/ocr-benchmark | tail -1
