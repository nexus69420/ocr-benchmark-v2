"""Score Chandra ocr_layout (v1 H100 outputs) vs v2 Gemini reference."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from ocr_benchmark_v2.src.metrics import anls, cer, length_ratio, wer
from ocr_benchmark_v2.src.normalize import normalize_model_output

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260618T103056Z"
CHANDRA = Path.home() / "ocr-benchmark/runs/chandra-2"

MAP = {
    ("Sabar_850_Milk_Rate_Gujarat", "page_001"): "gs_001",
    ("Sabar_850_Milk_Rate_Gujarat", "page_002"): "gs_002",
    ("Sabar_850_Milk_Rate_Gujarat", "page_003"): "gs_003",
    ("Sabar_850_Milk_Rate_Gujarat", "page_004"): "gs_004",
    ("DOC-20250210-WA0004", "page_001"): "gs_005",
    ("DOC-20250210-WA0004", "page_002"): "gs_006",
}


def main() -> None:
    rows = []
    for (doc, page), gid in MAP.items():
        ref = (RUN / "reference" / f"{gid}.txt").read_text(encoding="utf-8")
        ref_meta = json.loads((RUN / "reference" / f"{gid}.meta.json").read_text(encoding="utf-8"))
        md = (CHANDRA / doc / f"{page}.md").read_text(encoding="utf-8")
        hyp = normalize_model_output(md, strip_markdown=True)
        rows.append({
            "golden_id": gid,
            "doc_stem": doc,
            "page": page,
            "anls": round(anls(ref, hyp), 6),
            "cer": round(cer(ref, hyp), 6),
            "wer": round(wer(ref, hyp), 6),
            "reference_length": len(ref),
            "ocr_length": len(hyp),
            "length_ratio": round(length_ratio(ref, hyp), 6),
            "scoring_reliable": bool(ref_meta.get("scoring_reliable", True)),
        })
    summary = {
        "model": "chandra-2",
        "prompt_mode": "ocr_layout",
        "source": str(CHANDRA),
        "pages_scored": len(rows),
        "mean_anls": round(statistics.mean(r["anls"] for r in rows), 6),
        "median_anls": round(statistics.median(r["anls"] for r in rows), 6),
        "mean_cer": round(statistics.mean(r["cer"] for r in rows), 6),
        "mean_wer": round(statistics.mean(r["wer"] for r in rows), 6),
        "per_page": rows,
    }
    out = RUN / "scores/chandra_layout_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
