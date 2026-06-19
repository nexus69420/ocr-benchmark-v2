grep -n paddleocr-vl ~/amul-oan-api/ocr_benchmark_v2/config/models.yaml || echo MISSING
test -f ~/amul-oan-api/ocr_benchmark_v2/src/runners/paddle_runner.py && echo RUNNER_OK || echo RUNNER_MISSING
