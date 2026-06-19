source ~/ocr-benchmark/env.sh
cd ~/amul-oan-api
RUN=runs/20260618T103056Z
echo "=== chandra meta prompt_mode ==="
for f in $RUN/outputs/chandra-2/*.meta.json; do
  python -c "import json; d=json.load(open('$f')); print(d.get('golden_id'), d.get('prompt_mode'), d.get('normalized_char_count',0))"
done
echo "=== raw chandra backups? ==="
ls $RUN/raw/chandra-2/ 2>/dev/null | head -3
ls $RUN/outputs/chandra-2/ | wc -l
echo "=== any alternate score files ==="
ls $RUN/scores/
