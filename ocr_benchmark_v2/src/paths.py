"""Path constants for OCR benchmark v2."""

from __future__ import annotations

from pathlib import Path

# Package root: .../ocr_benchmark_v2/
V2_ROOT = Path(__file__).resolve().parents[1]
# Git repo root (parent of package)
REPO_ROOT = V2_ROOT.parent

GOLDEN_SET_ROOT = V2_ROOT / "golden_set"
GOLDEN_IMAGES = GOLDEN_SET_ROOT / "assets" / "images"
PROMPTS_DIR = V2_ROOT / "prompts"
CONFIG_DIR = V2_ROOT / "config"
RUNS_ROOT = REPO_ROOT / "runs"

DEFAULT_PROMPT_FILE = PROMPTS_DIR / "ocr_extract_v1.txt"
DEFAULT_MANIFEST = GOLDEN_SET_ROOT / "manifest.json"
