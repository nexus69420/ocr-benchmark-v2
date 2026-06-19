#!/usr/bin/env bash
# Install PaddleOCR-VL into the shared H100 venv (transformers engine).
# Models cache to $OCR_BENCH_ROOT/paddle-cache and HF hub on extended disk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/h100_env.sh"

echo "=== Installing paddleocr[doc-parser] (transformers backend) ==="
pip install -U "paddleocr[doc-parser]"

echo "=== Verify import ==="
python - <<'PY'
import os
os.chdir(os.path.expanduser("~/amul-oan-api"))
from paddleocr import PaddleOCRVL
print("PaddleOCRVL import ok", PaddleOCRVL)
PY

echo "=== Done ==="
pip show paddleocr | head -5
du -sh "$PADDLE_PDX_CACHE_HOME" 2>/dev/null || true
