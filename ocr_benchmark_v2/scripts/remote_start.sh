#!/bin/bash
set -e
ssh -o BatchMode=yes aicloud@10.185.25.197 <<'REMOTE'
find /home/aicloud/amul-oan-api/evals/ocr_benchmark_v2 -name '*.sh' -exec sed -i 's/\r$//' {} +
sed -i 's/\r$//' /tmp/launch_h100.sh
pkill -f launch_h100.sh 2>/dev/null || true
nohup /tmp/launch_h100.sh 20260618T103056Z > /tmp/ocr_v2_run.log 2>&1 &
sleep 12
tail -30 /tmp/ocr_v2_run.log
REMOTE
