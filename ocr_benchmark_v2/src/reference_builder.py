"""Build pseudo ground-truth reference texts from Gemini OCR outputs."""

from __future__ import annotations

import json
from typing import Any

from .config_loader import reference_model_id
from .dataset import load_pages
from .normalize import looks_truncated
from .run_context import RunContext


def build_reference(ctx: RunContext, *, force: bool = False) -> dict[str, Any]:
    ref_model = reference_model_id()
    ctx.ensure_dirs()
    built: list[dict[str, Any]] = []
    errors: list[str] = []

    for page in load_pages():
        gid = page.golden_id
        src = ctx.output_txt(ref_model, gid)
        dst = ctx.reference_txt(gid)
        meta_src = ctx.output_meta(ref_model, gid)

        if not force and dst.exists():
            built.append({"golden_id": gid, "status": "skipped", "reason": "exists"})
            continue

        if not src.exists():
            errors.append(f"{gid}: missing {ref_model} output {src}")
            continue

        text = src.read_text(encoding="utf-8").strip()
        ref_meta: dict[str, Any] = {}
        if meta_src.exists():
            ref_meta = json.loads(meta_src.read_text(encoding="utf-8"))

        truncated = bool(ref_meta.get("output_truncated")) or looks_truncated(text)
        empty = not bool(text)

        dst.write_text(text + "\n", encoding="utf-8")
        ref_meta_out = {
            "golden_id": gid,
            "source_model": ref_model,
            "source_meta": ref_meta,
            "reference_length": len(text),
            "reference_empty": empty,
            "reference_truncated": truncated,
            "scoring_reliable": not empty and not truncated,
        }
        (ctx.reference_dir / f"{gid}.meta.json").write_text(
            json.dumps(ref_meta_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        built.append(
            {
                "golden_id": gid,
                "status": "built",
                "reference_length": len(text),
                "reference_truncated": truncated,
                "scoring_reliable": ref_meta_out["scoring_reliable"],
            }
        )

    return {
        "run_id": ctx.run_id,
        "reference_model": ref_model,
        "built": built,
        "errors": errors,
        "ok": len(errors) == 0,
    }
