#!/bin/bash
set -e
ssh -o BatchMode=yes aicloud@10.185.25.197 <<'REMOTE'
find /home/aicloud/amul-oan-api/evals/ocr_benchmark_v2 -name '*.sh' -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' /tmp/launch_h100.sh
source /amulpfsdata/models/ocr-benchmark/env.sh
export CUDA_VISIBLE_DEVICES=6
export OCR_HF_LOCAL_ONLY=1
cd /home/aicloud/amul-oan-api
nohup bash -c '
  python -m ocr_benchmark_v2 run execute --run-id 20260618T103056Z --model chandra-2
  python -m ocr_benchmark_v2 run execute --run-id 20260618T103056Z --model qwen2.5-vl-7b
  python -m ocr_benchmark_v2 run score --run-id 20260618T103056Z
  python -m ocr_benchmark_v2 run report --run-id 20260618T103056Z
' > /tmp/ocr_v2_run.log 2>&1 &
sleep 5
tail -15 /tmp/ocr_v2_run.log
REMOTE
