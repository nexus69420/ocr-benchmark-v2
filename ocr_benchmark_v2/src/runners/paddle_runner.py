"""PaddleOCR-VL local document OCR runner (H100 GPU)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config_loader import load_models_config
from ..dataset import GoldenPage
from ..normalize import normalize_model_output
from ..paddle_env import apply_paddle_cache_env
from .types import OcrRunResult


def _extract_markdown_text(result: Any) -> str:
    md = getattr(result, "markdown", None)
    if isinstance(md, dict):
        text = md.get("markdown_texts", "")
    elif md is not None:
        text = md
    else:
        text = ""
    if isinstance(text, list):
        return "\n".join(str(t) for t in text).strip()
    return str(text or "").strip()


def _safe_json(result: Any) -> dict[str, Any]:
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        return data
    try:
        return json.loads(str(data))
    except (TypeError, json.JSONDecodeError):
        return {}


class PaddleRunner:
    """PaddleOCR-VL full pipeline (layout + VLM recognition)."""

    model_id = "paddleocr-vl"

    def __init__(self) -> None:
        cfg = load_models_config()["models"][self.model_id]
        self.variant = cfg.get("variant", "PaddleOCR-VL")
        self.engine = cfg.get("engine", "transformers")
        self.strip_markdown = bool(cfg.get("strip_markdown", True))
        self._pipeline = None
        self._env = apply_paddle_cache_env()

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        # Env must be set before paddleocr import (model cache path).
        from paddleocr import PaddleOCRVL

        kwargs: dict[str, Any] = {}
        if self.engine:
            kwargs["engine"] = self.engine
        self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def run_page(self, image_path: Path, page: GoldenPage) -> OcrRunResult:
        pipeline = self._load()
        t0 = time.perf_counter()
        outputs = list(pipeline.predict(str(image_path)))
        latency = time.perf_counter() - t0

        if not outputs:
            raise RuntimeError("PaddleOCR-VL returned no results")

        res = outputs[0]
        raw_text = _extract_markdown_text(res)
        normalized = normalize_model_output(raw_text, strip_markdown=self.strip_markdown)
        raw_json = _safe_json(res)

        return OcrRunResult(
            raw_text=raw_text,
            normalized_text=normalized,
            latency_sec=round(latency, 3),
            raw_payload={
                "variant": self.variant,
                "engine": self.engine,
                "paddle_cache": self._env,
                "json_keys": list(raw_json.keys()) if raw_json else [],
                "markdown_preview": raw_text[:2000],
            },
            meta={
                "runner_type": "paddle_vl",
                "variant": self.variant,
                "engine": self.engine,
                "paddle_pdx_cache_home": self._env.get("paddle_pdx_cache_home"),
                "output_empty": not bool(normalized),
                "raw_char_count": len(raw_text),
                "normalized_char_count": len(normalized),
            },
        )

    def close(self) -> None:
        self._pipeline = None
