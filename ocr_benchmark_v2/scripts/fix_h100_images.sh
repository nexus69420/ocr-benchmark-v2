#!/bin/bash
set -e
ssh -o BatchMode=yes aicloud@10.185.25.197 <<'REMOTE'
set -e
IMG_ROOT=/home/aicloud/amul-oan-api/ocr_benchmark_v2/golden_set/assets/images
SRC=/home/aicloud/ocr-benchmark/images
rm -rf "$IMG_ROOT"/*
mkdir -p "$IMG_ROOT"
cp -a "$SRC/Sabar_850_Milk_Rate_Gujarat" "$IMG_ROOT/"
cp -a "$SRC/DOC-20250210-WA0004" "$IMG_ROOT/"
ls -la "$IMG_ROOT/Sabar_850_Milk_Rate_Gujarat/"
ls -la "$IMG_ROOT/DOC-20250210-WA0004/"
python3 -m ocr_benchmark_v2 validate | tail -3
REMOTE
