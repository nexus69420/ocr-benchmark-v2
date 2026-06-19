source ~/ocr-benchmark/env.sh
cd ~/amul-oan-api
python -c "
import csv
from pathlib import Path
p=Path('runs/20260618T103056Z/scores/page_metrics.csv')
for r in csv.DictReader(p.open()):
    if r['model']=='paddleocr-vl':
        print(r['golden_id'], 'anls', r['anls'], 'cer', r['cer'], 'len', r['ocr_length'], 'ratio', r['length_ratio'])
"
