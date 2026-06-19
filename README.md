# Amul OCR Benchmark v2

Reproducible Gujarati scan OCR benchmark: frozen 6-page golden set, multi-model runners, ANLS/CER/WER scoring vs Gemini reference.

## Contents

| Path | Description |
|------|-------------|
| `ocr_benchmark_v2/` | Python package (CLI, runners, scorer, golden set images) |
| `runs/` | Benchmark run outputs (`runs/<run_id>/outputs`, `scores`, etc.) |
| `example.env` | API key template — copy to `.env` |
| `docs_*.md` | Comparison report and founder handoff notes |

## Quick start

```bash
git clone https://github.com/nexus69420/ocr-benchmark-v2.git
cd ocr-benchmark-v2
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # Linux
pip install -r ocr_benchmark_v2/requirements.txt
copy example.env .env         # add GEMINI_API_KEY, MISTRAL_API_KEY
```

```bash
# Validate golden set (6 PNGs + manifest)
python -m ocr_benchmark_v2 validate

# Create a versioned run
python -m ocr_benchmark_v2 run create

# Run OCR (API models from laptop)
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model gemini-ocr
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model mistral-ocr

# Build reference + score
python -m ocr_benchmark_v2 run reference --run-id <run_id>
python -m ocr_benchmark_v2 run score --run-id <run_id>
python -m ocr_benchmark_v2 run report --run-id <run_id>
```

GPU models (Chandra, Qwen, PaddleOCR-VL): see `ocr_benchmark_v2/README.md` and `ocr_benchmark_v2/scripts/`.

## Sample results

Run `20260618T103056Z` page-wise scores:

`runs/20260618T103056Z/scores/page_wise_breakdown.csv`

## Models

- **gemini-ocr** — reference pseudo-GT
- **mistral-ocr** — production candidate (API)
- **chandra-2** — self-hosted layout OCR (H100)
- **qwen2.5-vl-7b** — self-hosted VLM (H100)
- **paddleocr-vl** — PaddleOCR-VL-1.6 (H100)

## License

Internal Amul / OpenAgriNet evaluation tooling.
