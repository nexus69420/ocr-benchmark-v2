"""Hugging Face cache paths (H100: /amulpfsdata/models/ocr-benchmark)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_OCR_BENCH_ROOT = Path("/amulpfsdata/models/ocr-benchmark")


def ocr_bench_root() -> Path | None:
    raw = os.getenv("OCR_BENCH_ROOT")
    if raw:
        return Path(raw)
    if DEFAULT_OCR_BENCH_ROOT.is_dir():
        return DEFAULT_OCR_BENCH_ROOT
    return None


def apply_hf_cache_env() -> dict[str, str | None]:
    """Point HF caches at extended disk when OCR_BENCH_ROOT is set."""
    root = ocr_bench_root()
    applied: dict[str, str | None] = {
        "ocr_bench_root": str(root) if root else None,
        "hf_home": os.getenv("HF_HOME"),
        "hf_hub_cache": os.getenv("HF_HUB_CACHE"),
    }
    if root is None:
        return applied

    hf_home = os.getenv("HF_HOME") or str(root / "hf-cache")
    hf_hub = os.getenv("HF_HUB_CACHE") or str(Path(hf_home) / "hub")
    os.environ.setdefault("OCR_BENCH_ROOT", str(root))
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub)
    applied["hf_home"] = os.environ["HF_HOME"]
    applied["hf_hub_cache"] = os.environ["HF_HUB_CACHE"]
    return applied


def local_files_only() -> bool:
    return os.getenv("OCR_HF_LOCAL_ONLY", "1").lower() in ("1", "true", "yes")


def from_pretrained_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if local_files_only():
        kwargs["local_files_only"] = True
    return kwargs
