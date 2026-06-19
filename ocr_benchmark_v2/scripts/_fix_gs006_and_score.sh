source ~/ocr-benchmark/env.sh
IMG=~/amul-oan-api/ocr_benchmark_v2/golden_set/assets/images/DOC-20250210-WA0004/page_002.png
ls -la "$IMG" ~/ocr-benchmark/images/DOC-20250210-WA0004/page_002.png 2>&1
cp -a ~/ocr-benchmark/images/DOC-20250210-WA0004/page_002.png "$IMG"
python -c "from PIL import Image; Image.open('$IMG').verify(); print('png ok')"
cd ~/amul-oan-api
export CUDA_VISIBLE_DEVICES=6
export OCR_HF_LOCAL_ONLY=0
python -m ocr_benchmark_v2 run execute --run-id 20260618T103056Z --model paddleocr-vl
python -m ocr_benchmark_v2 run score --run-id 20260618T103056Z
python -m ocr_benchmark_v2 run report --run-id 20260618T103056Z
python -c "
import json
from pathlib import Path
s=json.loads(Path('runs/20260618T103056Z/scores/summary.json').read_text())
for m,v in s['models'].items():
    print(m, v.get('mean_anls'), v.get('mean_cer'), v.get('mean_wer'))
print('paddle', s['models'].get('paddleocr-vl'))
"
