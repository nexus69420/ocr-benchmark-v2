source ~/ocr-benchmark/env.sh
cd ~/amul-oan-api
python -m ocr_benchmark_v2 run plan --run-id 20260618T103056Z --model paddleocr-vl | head -30
