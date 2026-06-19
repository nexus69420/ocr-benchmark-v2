"""Write per-page OCR outputs to a run directory."""

from __future__ import annotations

import json
from typing import Any

from .dataset import GoldenPage
from .run_context import RunContext
from .runners.types import OcrRunResult


def _base_meta(model: str, golden_id: str, page: GoldenPage) -> dict[str, Any]:
    return {
        "model": model,
        "golden_id": golden_id,
        "doc_stem": page.doc_stem,
        "page": page.page,
        "image_path": str(page.image_path),
        "image_sha256": page.image_sha256,
    }


def write_unit_success(
    ctx: RunContext,
    model: str,
    golden_id: str,
    page: GoldenPage,
    result: OcrRunResult,
) -> None:
    txt_path = ctx.output_txt(model, golden_id)
    meta_path = ctx.output_meta(model, golden_id)
    raw_path = ctx.raw_json(model, golden_id)
    for p in (txt_path.parent, raw_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    txt_path.write_text(result.normalized_text + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps(result.raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "status": "ok",
        "latency_sec": result.latency_sec,
        **_base_meta(model, golden_id, page),
        **result.meta,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_unit_error(
    ctx: RunContext,
    model: str,
    golden_id: str,
    page: GoldenPage,
    error: str,
) -> None:
    meta_path = ctx.output_meta(model, golden_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "error",
        "error": error,
        **_base_meta(model, golden_id, page),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
