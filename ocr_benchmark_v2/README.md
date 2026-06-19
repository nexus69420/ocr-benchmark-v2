# OCR Benchmark v2

Reproducible Gujarati scan OCR benchmark framework.

## Milestones

| Milestone | Status |
|-----------|--------|
| M0 Scaffold | Done |
| M1 Dataset | Done |
| M2 Run ledger | Done |
| M3 Model runners | Done |
| M4 Reference builder | Done |
| M5 Metrics / scorer | Done |
| M6 Reports | Done |

## Quick start

```powershell
# Materialize golden set images (copies from v1)
python -m ocr_benchmark_v2 init

# Validate dataset (images + SHA256)
python -m ocr_benchmark_v2 validate

# Create a versioned run
python -m ocr_benchmark_v2 run create

# Show execution plan (resumable units)
python -m ocr_benchmark_v2 run plan --run-id <run_id>

# Run OCR (API models: gemini-ocr, mistral-ocr; GPU: chandra-2, qwen2.5-vl-7b)
python -m ocr_benchmark_v2 run execute --run-id <run_id>

# Single model only
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model gemini-ocr

# Build reference texts from Gemini outputs
python -m ocr_benchmark_v2 run reference --run-id <run_id>

# Score candidates vs reference
python -m ocr_benchmark_v2 run score --run-id <run_id>

# Markdown summary report
python -m ocr_benchmark_v2 run report --run-id <run_id>
```

## H100 (Chandra + Qwen) — models on extended disk

Weights live at `/amulpfsdata/models/ocr-benchmark/hf-cache/hub/` (symlinked `~/ocr-benchmark`).

**From laptop (sync + instructions):**
```powershell
.\evals\ocr_benchmark_v2\scripts\sync_v2_to_h100.ps1 -RunId 20260618T103056Z
```

**On H100:**
```bash
cd ~/amul-oan-api
export CUDA_VISIBLE_DEVICES=6   # free GPU
bash ocr_benchmark_v2/scripts/run_h100_pending.sh 20260618T103056Z
```

`h100_env.sh` sets `HF_HOME` / `HF_HUB_CACHE` to the extended disk and `OCR_HF_LOCAL_ONLY=1` (offline load).

**Pull results back:**
```powershell
scp -o ProxyJump=amul-vm3-uintele -r aicloud@10.185.25.197:~/amul-oan-api/runs/20260618T103056Z c:\amul-oan-api\eval_outputs\ocr_benchmark_v2\runs\
```

Runs live under `runs/<run_id>/` at the repo root.

v1 (`evals/ocr_benchmark/`) is **not modified**.
