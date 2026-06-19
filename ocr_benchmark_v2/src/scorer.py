"""Score candidate OCR outputs against reference."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .config_loader import candidate_model_ids, load_metrics_config, reference_model_id
from .dataset import load_pages
from .metrics import anls, cer, char_edit_details, length_ratio, levenshtein_distance, wer, word_edit_details
from .run_context import RunContext


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def score_run(ctx: RunContext, *, models: list[str] | None = None) -> dict[str, Any]:
    ctx.ensure_dirs()
    ref_model = reference_model_id()
    candidates = models or candidate_model_ids()
    pages = load_pages()
    rows: list[dict[str, Any]] = []

    for page in pages:
        gid = page.golden_id
        reference = _load_text(ctx.reference_txt(gid))
        ref_meta = _load_json(ctx.reference_dir / f"{gid}.meta.json")
        ref_truncated = bool(ref_meta.get("reference_truncated"))
        ref_empty = not bool(reference)
        scoring_reliable = bool(ref_meta.get("scoring_reliable", not ref_empty and not ref_truncated))

        for model in candidates:
            if model == ref_model:
                continue
            hyp_path = ctx.output_txt(model, gid)
            hyp_meta = _load_json(ctx.output_meta(model, gid))
            hypothesis = _load_text(hyp_path)

            if hyp_meta.get("status") == "error" or not hyp_path.exists():
                rows.append(
                    {
                        "run_id": ctx.run_id,
                        "golden_id": gid,
                        "doc_stem": page.doc_stem,
                        "page": page.page,
                        "model": model,
                        "status": "missing",
                        "scoring_reliable": False,
                    }
                )
                continue

            char_ed = char_edit_details(reference, hypothesis)
            word_ed = word_edit_details(reference, hypothesis)
            ref_len = len(reference)
            hyp_len = len(hypothesis)

            row: dict[str, Any] = {
                "run_id": ctx.run_id,
                "golden_id": gid,
                "doc_stem": page.doc_stem,
                "page": page.page,
                "model": model,
                "status": "scored",
                "anls": anls(reference, hypothesis),
                "cer": cer(reference, hypothesis),
                "wer": wer(reference, hypothesis),
                "levenshtein_distance": char_ed.distance,
                "insertions": char_ed.insertions,
                "deletions": char_ed.deletions,
                "substitutions": char_ed.substitutions,
                "word_insertions": word_ed.insertions,
                "word_deletions": word_ed.deletions,
                "word_substitutions": word_ed.substitutions,
                "reference_length": ref_len,
                "ocr_length": hyp_len,
                "length_ratio": length_ratio(reference, hypothesis),
                "latency_sec": hyp_meta.get("latency_sec"),
                "output_empty": not bool(hypothesis),
                "output_truncated": bool(hyp_meta.get("output_truncated")),
                "reference_truncated": ref_truncated,
                "reference_empty": ref_empty,
                "scoring_reliable": scoring_reliable,
                "continuation_passes": hyp_meta.get("continuation_passes"),
            }
            rows.append(row)

    csv_path = ctx.scores_dir / "page_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    summary = _aggregate(rows, candidates)
    summary_path = ctx.scores_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"run_id": ctx.run_id, "page_metrics_csv": str(csv_path), "summary": summary}


def _aggregate(rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    cfg = load_metrics_config()
    primary = cfg.get("primary", "anls")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "scored":
            continue
        by_model.setdefault(row["model"], []).append(row)

    model_stats: dict[str, Any] = {}
    for model in candidates:
        mrows = by_model.get(model, [])
        reliable = [r for r in mrows if r.get("scoring_reliable")]
        anls_vals = [r["anls"] for r in mrows if r.get("anls") is not None]
        rel_anls = [r["anls"] for r in reliable if r.get("anls") is not None]
        model_stats[model] = {
            "pages_scored": len(mrows),
            "pages_scoring_reliable": len(reliable),
            "mean_anls": round(statistics.mean(anls_vals), 6) if anls_vals else None,
            "median_anls": round(statistics.median(anls_vals), 6) if anls_vals else None,
            "mean_anls_reliable_only": round(statistics.mean(rel_anls), 6) if rel_anls else None,
            "mean_cer": round(statistics.mean(r["cer"] for r in mrows), 6) if mrows else None,
            "mean_wer": round(statistics.mean(r["wer"] for r in mrows), 6) if mrows else None,
            "mean_latency_sec": round(
                statistics.mean(r["latency_sec"] for r in mrows if r.get("latency_sec") is not None), 3
            )
            if any(r.get("latency_sec") is not None for r in mrows)
            else None,
        }

    ranked = sorted(
        ((m, s["mean_anls"]) for m, s in model_stats.items() if s.get("mean_anls") is not None),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "primary_metric": primary,
        "models": model_stats,
        "ranking_by_mean_anls": [{"model": m, "mean_anls": v} for m, v in ranked],
    }
