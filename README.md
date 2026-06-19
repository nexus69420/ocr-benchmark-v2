# OCR Benchmark v2

Reproducible benchmark for **Gujarati scanned-document OCR**. Compares multiple OCR engines on a **frozen 6-page golden set**, scores them against a **Gemini-generated reference**, and optionally runs a **Gemini image-grounded judge**.

**Repo:** [github.com/nexus69420/ocr-benchmark-v2](https://github.com/nexus69420/ocr-benchmark-v2)

---

## Table of contents

1. [What problem this solves](#what-problem-this-solves)
2. [High-level pipeline](#high-level-pipeline)
3. [Golden set (frozen inputs)](#golden-set-frozen-inputs)
4. [Models and how they run](#models-and-how-they-run)
5. [Run directory layout](#run-directory-layout)
6. [CLI phases (step by step)](#cli-phases-step-by-step)
7. [Reference text (pseudo ground truth)](#reference-text-pseudo-ground-truth)
8. [Text normalization before scoring](#text-normalization-before-scoring)
9. [Metrics explained](#metrics-explained)
10. [Image-grounded judge (optional)](#image-grounded-judge-optional)
11. [Resumability and idempotency](#resumability-and-idempotency)
12. [Quick start](#quick-start)
13. [Sample results](#sample-results)
14. [Project layout](#project-layout)

---

## What problem this solves

Amul ingests Gujarati agricultural PDFs (milk-rate circulars, vet notices, etc.). We need to pick and monitor an OCR engine that:

- Keeps **Gujarati in Gujarati script** (not Devanagari/Hindi)
- Extracts **tables and dense text** faithfully
- Can be **compared fairly** across API models (Mistral, Gemini) and self-hosted GPU models (Chandra, Qwen, PaddleOCR)

v2 replaces ad-hoc v1 scripts with a **versioned, resumable pipeline**: same images every run, same metrics, auditable outputs.

---

## High-level pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GOLDEN SET (frozen, never changes between runs)                        │
│  6 PNG page images + manifest.json (SHA256 per image)                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: run create                                                    │
│  Create runs/<run_id>/ + config.snapshot.json (prompt hash, models…)   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: run execute  (per model × per page)                           │
│                                                                         │
│   page PNG  ──►  OCR runner  ──►  outputs/<model>/gs_XXX.txt            │
│                                   + .meta.json (latency, flags)         │
│                                   + raw/<model>/gs_XXX.json (API dump)  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: run reference                                                 │
│  Copy Gemini outputs ──►  reference/gs_XXX.txt  (pseudo-GT)             │
│  Flag truncated references (scoring_reliable=false)                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: run score                                                     │
│  Compare each candidate .txt vs reference/  ──►  page_metrics.csv       │
│                                                  summary.json           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: run report                                                    │
│  Human-readable scores/REPORT.md from summary.json                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼ (optional)
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: run judge                                                     │
│  Gemini reads page image + OCR text  ──►  judge/<model>/gs_XXX.json     │
│  (qualitative score 0–100, error lists)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Unit of work:** one `(run_id, model, golden_id)` tuple — e.g. `mistral-ocr` on `gs_002`.

**Primary ranking metric:** **mean ANLS** (see [Metrics explained](#metrics-explained)).

---

## Golden set (frozen inputs)

| Golden ID | Document | Page | Content type |
|-----------|----------|------|--------------|
| gs_001 | Sabar milk-rate circular | page_001 | Header + intro tables |
| gs_002 | Sabar milk-rate circular | page_002 | Dense rate table |
| gs_003 | Sabar milk-rate circular | page_003 | Dense rate table |
| gs_004 | Sabar milk-rate circular | page_004 | Dense rate table |
| gs_005 | DOC WhatsApp scan | page_001 | Multi-column notice |
| gs_006 | DOC WhatsApp scan | page_002 | Multi-column notice |

**Images live at:**

```
ocr_benchmark_v2/golden_set/assets/images/
  Sabar_850_Milk_Rate_Gujarat/page_001.png … page_004.png
  DOC-20250210-WA0004/page_001.png, page_002.png
```

**Manifest:** `ocr_benchmark_v2/golden_set/manifest.json` — lists every page, image path, SHA256, tags.

Validate before any run:

```bash
python -m ocr_benchmark_v2 validate
```

This checks all 6 PNGs exist and match their hashes. The golden set is **intentionally small and hard** (table-heavy Gujarati scans) — results are directional, not a full corpus proof.

---

## Models and how they run

Configured in `ocr_benchmark_v2/config/models.yaml`.

| Model | Where it runs | Prompt | Markdown strip on save? | Role |
|-------|---------------|--------|-------------------------|------|
| **gemini-ocr** | Laptop (API) | Shared `ocr_extract_v1.txt` + continuation if truncated | No (plain text) | Reference + candidate |
| **mistral-ocr** | Laptop (API) | Native Mistral OCR API | **Yes** | Production candidate |
| **qwen2.5-vl-7b** | H100 GPU | Shared `ocr_extract_v1.txt` | No | Self-hosted candidate |
| **chandra-2** | H100 GPU | Native `ocr_layout` (recommended) or shared prompt (not recommended) | Layout: strip at score time | Self-hosted candidate |
| **paddleocr-vl** | H100 GPU | PaddleOCR-VL native | **Yes** | Exploratory (poor Gujarati) |

**Shared prompt** (`ocr_benchmark_v2/prompts/ocr_extract_v1.txt`): instructs the VLM to output **plain Gujarati/English text only** — no markdown, no translation, no summarization.

**Fairness note:** Mistral and Paddle emit markdown/HTML tables. We strip formatting tokens (`#`, `|`, `**`, HTML tags) before saving or scoring so we compare **text content**, not serialization format. Gemini and Qwen already output plain text. Chandra layout outputs markdown tables and needs the same strip when scored (see `page_wise_breakdown.csv` in sample results).

GPU runners and H100 setup: `ocr_benchmark_v2/scripts/` (see `run_h100_pending.sh`, `h100_env.sh`).

---

## Run directory layout

Each benchmark execution gets a unique **`run_id`** (UTC timestamp, e.g. `20260618T103056Z`).

```
runs/<run_id>/
├── config.snapshot.json     # frozen config: models, prompt hash, golden manifest hash
├── ledger.jsonl             # append-only log of every (page, model) attempt
│
├── outputs/               # normalized OCR text per model
│   ├── gemini-ocr/
│   │   ├── gs_001.txt
│   │   ├── gs_001.meta.json    # latency, truncation flags, char counts
│   │   └── …
│   ├── mistral-ocr/
│   ├── chandra-2/
│   ├── qwen2.5-vl-7b/
│   └── paddleocr-vl/
│
├── raw/                   # optional raw API / model payloads (audit)
│   └── <model>/gs_XXX.json
│
├── reference/             # pseudo ground truth (from Gemini)
│   ├── gs_001.txt
│   ├── gs_001.meta.json    # reference_truncated, scoring_reliable
│   └── …
│
├── scores/
│   ├── page_metrics.csv    # one row per (page × model) — full detail
│   ├── summary.json        # mean ANLS, CER, WER per model
│   └── REPORT.md           # human summary
│
└── judge/                  # optional Phase 6
    ├── mistral-ocr/gs_001.json
    ├── judge_scores.csv
    └── judge_rollup.json
```

**Separation of concerns:**

- `outputs/` = what each model said (after per-model normalization)
- `reference/` = what we compare against (Gemini copy)
- `scores/` = automated string metrics
- `judge/` = qualitative vision-based review

---

## CLI phases (step by step)

All commands run from the **repo root** (parent of `ocr_benchmark_v2/`).

### 0. One-time setup

```bash
git clone https://github.com/nexus69420/ocr-benchmark-v2.git
cd ocr-benchmark-v2
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r ocr_benchmark_v2/requirements.txt
copy example.env .env           # add GEMINI_API_KEY, MISTRAL_API_KEY
python -m ocr_benchmark_v2 validate
```

### 1. `run create` — allocate a run folder

```bash
python -m ocr_benchmark_v2 run create
# → {"run_id": "20260618T103056Z", "run_root": "runs/20260618T103056Z"}
```

Writes `config.snapshot.json` so you can reproduce exactly which prompt, models, and golden manifest were used.

### 2. `run plan` — preview work units

```bash
python -m ocr_benchmark_v2 run plan --run-id <run_id>
```

Shows pending vs skipped units. A unit is **skipped** if `outputs/<model>/gs_XXX.txt` already exists (resumable).

### 3. `run execute` — run OCR

```bash
# All models (API + GPU if configured)
python -m ocr_benchmark_v2 run execute --run-id <run_id>

# Single model
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model gemini-ocr
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model mistral-ocr
```

For each pending `(model, page)`:

1. Load page PNG from golden set
2. Call the model runner (`src/runners/*.py`)
3. Normalize text (strip markdown if configured)
4. Write `outputs/<model>/gs_XXX.txt` + `.meta.json`
5. Append event to `ledger.jsonl`

**Typical order:** run `gemini-ocr` first (needed for reference), then candidates.

### 4. `run reference` — build pseudo ground truth

```bash
python -m ocr_benchmark_v2 run reference --run-id <run_id>
```

Copies `outputs/gemini-ocr/gs_XXX.txt` → `reference/gs_XXX.txt`.

Sets flags per page:

- `reference_truncated=true` → Gemini output was cut off (continuation helps but gs_005 may still truncate)
- `scoring_reliable=false` → down-weight that page in rankings

The reference is **not human-verified truth** — the scanned image is ultimate truth. The judge phase exists for disputed pages.

### 5. `run score` — compute metrics

```bash
python -m ocr_benchmark_v2 run score --run-id <run_id>
```

For every candidate model (excludes gemini-ocr), for every page:

- Load `reference/gs_XXX.txt` and `outputs/<model>/gs_XXX.txt`
- Compute ANLS, CER, WER, Levenshtein, insert/delete/substitute counts
- Write `scores/page_metrics.csv` and aggregate `scores/summary.json`

### 6. `run report` — markdown summary

```bash
python -m ocr_benchmark_v2 run report --run-id <run_id>
```

Writes `scores/REPORT.md` with model ranking by mean ANLS.

### 7. `run judge` — optional qualitative review

```bash
python -m ocr_benchmark_v2 run judge --run-id <run_id> --export-csv
```

Sends each page image + candidate OCR text to **Gemini 2.5 Flash** with a structured judge prompt. Returns scores 0–100, lists of missing/hallucinated content, layout issues.

Use when string metrics disagree with visual quality, or when reference is imperfect.

---

## Reference text (pseudo ground truth)

**Why Gemini?** Fast, multilingual, vision-capable, and we control the same shared extraction prompt as Qwen.

**Continuation:** If Gemini hits `max_tokens`, the runner automatically re-prompts with “continue where you stopped” (up to 3 passes). This matters on dense milk-rate tables where a single pass used to truncate at ~500 characters.

**Limitation:** `gs_005` may still have `reference_truncated=true`. Scores on that page are computed but flagged `scoring_reliable=false` in `page_metrics.csv`.

---

## Text normalization before scoring

Goal: compare **extracted characters**, not markdown syntax.

Applied in `src/normalize.py` when each runner saves output (and again identically for reference at score time).

| Step | What it does |
|------|----------------|
| Markdown strip (Mistral, Paddle, Chandra layout) | Remove `#`, `**`, `\|`, HTML tags, code fences — keep words/numbers |
| Unicode NFC | Canonical Gujarati composed form |
| Zero-width strip | Remove BOM, `\u200b`, etc. |
| Preserve newlines | Line breaks kept |
| Collapse horizontal whitespace | Multiple spaces/tabs → single space per line |

**What we do NOT do:** spell correction, Gujarati↔ASCII digit normalization, translation, or table reconstruction.

See `ocr_benchmark_v2/NORMALIZATION.md` for the spec.

**Why this matters:** Chandra layout without markdown strip scored ~0.33 mean ANLS; with fair strip ~0.79 — same content, different formatting penalty.

---

## Metrics explained

### ANLS (primary)

**Average Normalized Levenshtein Similarity** — primary ranking column.

```
similarity = 1 - (edit_distance / max(len(reference), len(hypothesis)))
ANLS = similarity if similarity >= 0.5 else 0
```

- Range per page: 0.0 to 1.0 (higher = closer to reference)
- The **0.5 threshold** means small differences pass through; large divergences become zero
- Harsher than v1’s `difflib.SequenceMatcher` ratio

### CER / WER (secondary)

- **CER** = character edit distance / reference length (lower = better)
- **WER** = word edit distance / reference word count (lower = better)

### Length ratio

`ocr_length / reference_length` — sanity check for over-generation (Chandra shared-prompt bug hit 1.8–2.9×) or under-extraction.

### Edit breakdown (`page_metrics.csv`)

| Column | Meaning |
|--------|---------|
| Insert | Characters in OCR but not in reference |
| Delete | Characters in reference but missing from OCR |
| Substitute | Wrong character replacements |

### `scoring_reliable`

`false` when reference was truncated or empty — still scored, but exclude from “reliable only” means when interpreting rankings.

---

## Image-grounded judge (optional)

Automatic metrics compare strings. They fail when:

- Reference is incomplete but OCR captured more (looks “wrong” but is better)
- Layout differs but content is correct
- You need a human-like verdict without manual Gujarati review

The judge (`src/image_judge.py`) sends **the page image + OCR text** to Gemini with a fixed rubric:

- Overall score 0–100
- Missing / hallucinated content lists
- Character accuracy estimate
- Layout issues

Prompt: `ocr_benchmark_v2/judge/gemini_judge_prompt.py`

**Cost:** one Gemini API call per `(page, model)` — 6 pages × 4 candidates = 24 calls per full judge run.

---

## Resumability and idempotency

| Feature | How |
|---------|-----|
| **Skip completed work** | If `outputs/<model>/gs_XXX.txt` exists → unit skipped on re-execute |
| **Force re-run** | `--force` on plan/execute/judge |
| **Ledger** | `ledger.jsonl` records every attempt, error, latency |
| **Immutable runs** | Never overwrite `run_id`; create a new run to change prompt/models |
| **Re-score without re-OCR** | `run score` only reads `outputs/` + `reference/` |

GPU runs on H100 can take hours; resumability is essential.

---

## Quick start

Full end-to-end (API models only, from laptop):

```bash
python -m ocr_benchmark_v2 validate
python -m ocr_benchmark_v2 run create
# note the run_id printed

python -m ocr_benchmark_v2 run execute --run-id <run_id> --model gemini-ocr
python -m ocr_benchmark_v2 run execute --run-id <run_id> --model mistral-ocr
python -m ocr_benchmark_v2 run reference --run-id <run_id>
python -m ocr_benchmark_v2 run score --run-id <run_id>
python -m ocr_benchmark_v2 run report --run-id <run_id>
```

List all runs:

```bash
python -m ocr_benchmark_v2 run list
```

---

## Sample results

Run `20260618T103056Z` (Amul pilot, June 2026):

**Page-wise breakdown (founder CSV):**

`runs/20260618T103056Z/scores/page_wise_breakdown.csv`

| Model | Mean ANLS | Notes |
|-------|-----------|-------|
| Mistral | 0.799 | Production pick — fast (~3 s/page), best overall |
| Chandra (layout, fair strip) | 0.791 | Competitive accuracy; ~150 s/page on H100 |
| Qwen 7B | 0.407 | Fails on dense tables |
| PaddleOCR-VL | 0.000 | Wrong script (Devanagari) for Gujarati |

**Reports:**

- `docs_FINAL_COMPARISON_REPORT.md` — full v1 vs v2 analysis

---

## Project layout

```
ocr-benchmark-v2/                    ← repo root (you are here)
├── README.md                        ← this file
├── example.env                      ← API keys template
├── runs/                            ← benchmark outputs (gitignored except samples)
├── docs_*.md                        ← reports
│
└── ocr_benchmark_v2/                ← Python package
    ├── cli.py                       ← CLI entry (python -m ocr_benchmark_v2)
    ├── config/
    │   ├── models.yaml              ← model registry
    │   ├── metrics.yaml             ← primary metric = anls
    │   └── rate_limits.yaml         ← API throttling
    ├── golden_set/
    │   ├── manifest.json
    │   └── assets/images/           ← 6 frozen PNGs
    ├── prompts/
    │   └── ocr_extract_v1.txt       ← shared VLM extraction prompt
    ├── judge/
    │   └── gemini_judge_prompt.py
    ├── src/
    │   ├── orchestrator.py          ← execute loop
    │   ├── scorer.py                ← page_metrics.csv
    │   ├── metrics.py               ← ANLS, CER, WER
    │   ├── normalize.py             ← text normalization
    │   ├── reference_builder.py
    │   ├── image_judge.py
    │   └── runners/                 ← one file per OCR engine
    └── scripts/                     ← H100 sync & GPU helpers
```

---

## v1 vs v2 (one paragraph)

**v1** (`evals/ocr_benchmark` in amul-oan-api): 73 pages, script-gate (≥95% Gujarati purity), `difflib` similarity vs Gemini, ad-hoc CSV exports.

**v2** (this repo): 6 frozen golden pages, versioned runs, **ANLS** primary metric, continuation-aware Gemini reference, per-page edit breakdown, optional image judge, resumable CLI.

---

## License

Internal Amul / OpenAgriNet evaluation tooling.
