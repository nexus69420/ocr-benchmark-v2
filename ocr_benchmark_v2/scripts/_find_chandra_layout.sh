source ~/ocr-benchmark/env.sh
for stem in Sabar_850_Milk_Rate_Gujarat DOC-20250210-WA0004; do
  echo "=== $stem ==="
  ls ~/ocr-benchmark/runs/chandra-2/$stem/ 2>/dev/null | head -6
done
# v2 run on h100
ls ~/amul-oan-api/runs/20260618T103056Z/outputs/chandra-2/ | head -3
