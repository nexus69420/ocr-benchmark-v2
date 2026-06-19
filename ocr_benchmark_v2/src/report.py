"""Generate human-readable benchmark report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run_context import RunContext


def write_report(ctx: RunContext) -> dict[str, Any]:
    summary_path = ctx.scores_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run scores first: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        f"# OCR Benchmark v2 Report — `{ctx.run_id}`",
        "",
        f"**Primary metric:** {summary.get('primary_metric', 'anls')}",
        "",
        "## Model summary (mean ANLS, all pages)",
        "",
        "| Model | Mean ANLS | Mean CER | Mean WER | Reliable pages | Latency (s) |",
        "|-------|-----------|----------|----------|----------------|-------------|",
    ]

    for model, stats in summary.get("models", {}).items():
        lines.append(
            f"| {model} | {stats.get('mean_anls', '—')} | {stats.get('mean_cer', '—')} | "
            f"{stats.get('mean_wer', '—')} | {stats.get('pages_scoring_reliable', 0)}/"
            f"{stats.get('pages_scored', 0)} | {stats.get('mean_latency_sec', '—')} |"
        )

    lines.extend(["", "## Ranking", ""])
    for i, row in enumerate(summary.get("ranking_by_mean_anls", []), 1):
        lines.append(f"{i}. **{row['model']}** — ANLS {row['mean_anls']}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Pages with `reference_truncated=true` are scored but marked `scoring_reliable=false`.",
            "- See `scores/page_metrics.csv` for per-page detail.",
            "",
        ]
    )

    report_path = ctx.scores_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"run_id": ctx.run_id, "report": str(report_path)}
