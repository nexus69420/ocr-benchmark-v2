#!/usr/bin/env python3
"""Score Chandra ocr_layout outputs (v1 paths on H100) vs v2 Gemini reference."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs/20260618T103056Z"
sys.path.insert(0, str(REPO / "evals/ocr_benchmark_v2"))

from src.dataset import load_pages  # noqa: E402
from src.metrics import anls, cer, length_ratio, wer  # noqa: E402
from src.normalize import normalize_model_output  # noqa: E402

GOLDEN_CHANDRA = {
    ("Sabar_850_Milk_Rate_Gujarat", "page_001"): "gs_001",
    ("Sabar_850_Milk_Rate_Gujarat", "page_002"): "gs_002",
    ("Sabar_850_Milk_Rate_Gujarat", "page_003"): "gs_003",
    ("Sabar_850_Milk_Rate_Gujarat", "page_004"): "gs_004",
    ("DOC-20250210-WA0004", "page_001"): "gs_005",
    ("DOC-20250210-WA0004", "page_002"): "gs_006",
}


def pull_chandra_md() -> Path:
  with tempfile.TemporaryDirectory() as td:
    tgz = Path(td) / "chandra_layout.tgz"
    remote = "ssh -o BatchMode=yes amul-vm5-ai-backend ssh -o BatchMode=yes aicloud@10.185.25.197 bash -s"
    script = """
set -e
cd ~/ocr-benchmark/runs/chandra-2
tar czf /tmp/chandra_golden.tgz Sabar_850_Milk_Rate_Gujarat DOC-20250210-WA0004
"""
    subprocess.run(remote.split(), input=script.encode(), check=True)
    subprocess.run(
      ["ssh", "-o", "BatchMode=yes", "amul-vm5-ai-backend",
       "scp", "-o", "BatchMode=yes", "aicloud@10.185.25.197:/tmp/chandra_golden.tgz", str(tgz)],
      check=True,
    )
    out = RUN / "chandra_layout_v1"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tar:
      tar.extractall(out)
    return out


def main() -> None:
  root = pull_chandra_md()
  rows = []
  for page in load_pages():
    key = (page.doc_stem, page.page)
    gid = GOLDEN_CHANDRA[key]
    md = root / page.doc_stem / f"{page.page}.md"
    ref = (RUN / "reference" / f"{gid}.txt").read_text(encoding="utf-8")
    ref_meta = json.loads((RUN / "reference" / f"{gid}.meta.json").read_text(encoding="utf-8"))
    hyp = normalize_model_output(md.read_text(encoding="utf-8"), strip_markdown=True)
    reliable = bool(ref_meta.get("scoring_reliable", True))
    rows.append({
      "golden_id": gid,
      "doc_stem": page.doc_stem,
      "page": page.page,
      "anls": anls(ref, hyp),
      "cer": cer(ref, hyp),
      "wer": wer(ref, hyp),
      "reference_length": len(ref),
      "ocr_length": len(hyp),
      "length_ratio": length_ratio(ref, hyp),
      "scoring_reliable": reliable,
    })
  anls_vals = [r["anls"] for r in rows]
  summary = {
    "model": "chandra-2",
    "prompt_mode": "ocr_layout",
    "note": "Native Chandra layout mode (v1 runner on H100); not shared-prompt re-run",
    "pages_scored": len(rows),
    "pages_scoring_reliable": sum(1 for r in rows if r["scoring_reliable"]),
    "mean_anls": round(statistics.mean(anls_vals), 6),
    "median_anls": round(statistics.median(anls_vals), 6),
    "mean_cer": round(statistics.mean(r["cer"] for r in rows), 6),
    "mean_wer": round(statistics.mean(r["wer"] for r in rows), 6),
    "per_page": rows,
  }
  out = RUN / "scores" / "chandra_layout_summary.json"
  out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
