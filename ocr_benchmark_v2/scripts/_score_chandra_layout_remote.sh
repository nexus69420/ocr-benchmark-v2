#!/bin/bash
set -euo pipefail
cd ~/amul-oan-api
source /amulpfsdata/models/ocr-benchmark/env.sh
python3 -m ocr_benchmark_v2.scripts.score_chandra_layout_h100
