"""Validate / refresh golden set manifest (images ship with the repo)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import DEFAULT_MANIFEST, GOLDEN_IMAGES, GOLDEN_SET_ROOT, REPO_ROOT

GOLDEN_SET_ID = "ocr_gujarati_scan_v1"
VERSION = "1.0.0"
HISTORICAL_TRUNCATED = {"gs_002", "gs_003", "gs_004"}

PAGE_SPECS = [
    ("gs_001", "Sabar_850_Milk_Rate_Gujarat", "Sabar_850 Milk Rate Gujarat.pdf", "page_001", 0, ["milk_rate", "header", "intro_tables"], False),
    ("gs_002", "Sabar_850_Milk_Rate_Gujarat", "Sabar_850 Milk Rate Gujarat.pdf", "page_002", 1, ["milk_rate", "dense_table"], True),
    ("gs_003", "Sabar_850_Milk_Rate_Gujarat", "Sabar_850 Milk Rate Gujarat.pdf", "page_003", 2, ["milk_rate", "dense_table"], True),
    ("gs_004", "Sabar_850_Milk_Rate_Gujarat", "Sabar_850 Milk Rate Gujarat.pdf", "page_004", 3, ["milk_rate", "dense_table"], True),
    ("gs_005", "DOC-20250210-WA0004", "DOC-20250210-WA0004.pdf", "page_001", 0, ["vet_products", "multi_column"], False),
    ("gs_006", "DOC-20250210-WA0004", "DOC-20250210-WA0004.pdf", "page_002", 1, ["vet_products", "multi_column"], False),
]


def build_manifest() -> dict:
    pages = []
    for gid, stem, pdf_name, page, page_index, tags, hist_trunc in PAGE_SPECS:
        rel_img = GOLDEN_IMAGES.relative_to(REPO_ROOT) / stem / f"{page}.png"
        img = GOLDEN_IMAGES / stem / f"{page}.png"
        if not img.exists():
            raise FileNotFoundError(f"Missing golden image: {img}")
        pages.append(
            {
                "golden_id": gid,
                "doc_stem": stem,
                "pdf_name": pdf_name,
                "pdf_path": f"source_pdfs/{pdf_name}",
                "page": page,
                "page_index": page_index,
                "language": "gu",
                "doc_type": "scanned_circular" if "Sabar" in stem else "whatsapp_scan",
                "tags": tags,
                "image_path": str(rel_img).replace("\\", "/"),
                "historical_reference_truncated": hist_trunc,
            }
        )
    return {
        "golden_set_id": GOLDEN_SET_ID,
        "version": VERSION,
        "description": "6-page Gujarati scan OCR golden set (milk rate p1-4 + DOC p1-2)",
        "page_count": len(pages),
        "source": "amul-ocr-benchmark-v2",
        "pages": pages,
    }


def write_csv(manifest: dict, path: Path) -> None:
    fields = [
        "golden_id", "doc_stem", "pdf_name", "page", "page_index", "language",
        "doc_type", "tags", "image_path", "pdf_path", "historical_reference_truncated",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in manifest["pages"]:
            w.writerow({**p, "tags": ",".join(p["tags"])})


def main() -> None:
    GOLDEN_SET_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(manifest, GOLDEN_SET_ROOT / "golden_set.csv")
    print(f"Wrote {DEFAULT_MANIFEST} ({manifest['page_count']} pages)")
    print(f"Images: {GOLDEN_IMAGES}")


if __name__ == "__main__":
    main()
