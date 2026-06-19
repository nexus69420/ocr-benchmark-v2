"""PaddleOCR / PaddleX model cache on extended disk (H100)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .hf_env import apply_hf_cache_env, ocr_bench_root


def apply_paddle_cache_env() -> dict[str, Any]:
    """Set PaddleX cache + HF dirs before importing paddleocr."""
    hf = apply_hf_cache_env()
    root = ocr_bench_root()
    applied: dict[str, Any] = {"ocr_bench_root": hf.get("ocr_bench_root"), **hf}

    if root is not None:
        paddle_cache = os.getenv("PADDLE_PDX_CACHE_HOME") or str(root / "paddle-cache")
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", paddle_cache)
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        applied["paddle_pdx_cache_home"] = os.environ["PADDLE_PDX_CACHE_HOME"]
        applied["paddle_model_source"] = os.environ.get("PADDLE_PDX_MODEL_SOURCE")

    return applied
