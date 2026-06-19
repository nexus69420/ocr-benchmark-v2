"""Golden set dataset loader and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import DEFAULT_MANIFEST, GOLDEN_IMAGES, REPO_ROOT


@dataclass(frozen=True)
class GoldenPage:
    golden_id: str
    doc_stem: str
    pdf_name: str
    page: str
    page_index: int
    language: str
    doc_type: str
    tags: list[str]
    image_path: Path
    pdf_path: Path
    image_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> int:
        return self.page_index + 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_repo_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Golden manifest not found: {path}. Run: python -m ocr_benchmark_v2 init"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_sha256(path: Path | None = None) -> str:
    path = path or DEFAULT_MANIFEST
    return _sha256_file(path)


def load_pages(manifest: dict[str, Any] | None = None, *, compute_hash: bool = True) -> list[GoldenPage]:
    manifest = manifest or load_manifest()
    pages: list[GoldenPage] = []
    for row in manifest.get("pages", []):
        img = _resolve_repo_path(row["image_path"])
        pdf = _resolve_repo_path(row["pdf_path"])
        sha = _sha256_file(img) if compute_hash and img.exists() else None
        extra = {k: v for k, v in row.items() if k not in {
            "golden_id", "doc_stem", "pdf_name", "page", "page_index", "language",
            "doc_type", "tags", "image_path", "pdf_path",
        }}
        pages.append(
            GoldenPage(
                golden_id=row["golden_id"],
                doc_stem=row["doc_stem"],
                pdf_name=row["pdf_name"],
                page=row["page"],
                page_index=int(row["page_index"]),
                language=row.get("language", "gu"),
                doc_type=row.get("doc_type", "unknown"),
                tags=list(row.get("tags", [])),
                image_path=img,
                pdf_path=pdf,
                image_sha256=sha,
                extra=extra,
            )
        )
    return pages


def validate_dataset(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or load_manifest()
    pages = load_pages(manifest, compute_hash=True)
    errors: list[str] = []
    warnings: list[str] = []

    for p in pages:
        if not p.image_path.exists():
            errors.append(f"{p.golden_id}: missing image {p.image_path}")
        if not p.pdf_path.exists():
            warnings.append(f"{p.golden_id}: missing PDF {p.pdf_path}")
        if p.extra.get("historical_reference_truncated"):
            warnings.append(f"{p.golden_id}: historically known truncated Gemini reference page")

    return {
        "golden_set_id": manifest.get("golden_set_id"),
        "version": manifest.get("version"),
        "page_count": len(pages),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "pages": [
            {
                "golden_id": p.golden_id,
                "image_path": str(p.image_path),
                "image_sha256": p.image_sha256,
                "doc_stem": p.doc_stem,
                "page": p.page,
            }
            for p in pages
        ],
    }


def default_image_path(doc_stem: str, page: str, images_root: Path | None = None) -> Path:
    root = images_root or GOLDEN_IMAGES
    return root / doc_stem / f"{page}.png"
