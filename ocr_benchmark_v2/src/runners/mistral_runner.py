"""Mistral OCR API runner."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx

from ..config_loader import load_models_config, load_rate_limits
from ..dataset import GoldenPage
from ..normalize import normalize_model_output
from .retry import RateLimiter, call_with_retry
from .types import OcrRunResult


def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class MistralRunner:
    model_id = "mistral-ocr"

    def __init__(self) -> None:
        cfg = load_models_config()["models"][self.model_id]
        limits = load_rate_limits().get(self.model_id, {})
        self.api_model = os.getenv(cfg["api_model_env"], cfg["default_api_model"])
        self.strip_markdown = bool(cfg.get("strip_markdown", True))
        self.table_format = os.getenv("MISTRAL_TABLE_FORMAT", "") or None
        self._api_key = os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set MISTRAL_API_KEY in .env")
        self._limiter = RateLimiter(float(limits.get("min_interval_sec", 0)))
        self._max_retries = int(limits.get("max_retries", 5))
        self._backoff_base = float(limits.get("backoff_base_sec", 2))
        self._backoff_max = float(limits.get("backoff_max_sec", 30))
        self._client = httpx.Client(timeout=600.0)

    def run_page(self, image_path: Path, page: GoldenPage) -> OcrRunResult:
        payload: dict = {
            "model": self.api_model,
            "document": {"type": "image_url", "image_url": _image_to_data_url(image_path)},
        }
        if self.table_format:
            payload["table_format"] = self.table_format

        self._limiter.wait()
        t0 = time.perf_counter()

        def _call():
            r = self._client.post(
                "https://api.mistral.ai/v1/ocr",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            r.raise_for_status()
            return r.json()

        data = call_with_retry(
            _call,
            max_retries=self._max_retries,
            backoff_base_sec=self._backoff_base,
            backoff_max_sec=self._backoff_max,
            on_retry=lambda attempt, err, wait: print(
                f"  mistral retry {attempt}/{self._max_retries} in {wait:.0f}s ({err})",
                flush=True,
            ),
        )
        latency = time.perf_counter() - t0
        self._limiter.mark()

        pages = data.get("pages") or []
        if not pages:
            raise RuntimeError(f"No pages in Mistral OCR response: {data}")
        raw_text = (pages[0].get("markdown") or "").strip()
        normalized = normalize_model_output(raw_text, strip_markdown=self.strip_markdown)

        return OcrRunResult(
            raw_text=raw_text,
            normalized_text=normalized,
            latency_sec=round(latency, 3),
            raw_payload={"api_model": self.api_model, "request": payload, "response": data},
            meta={
                "runner_type": "api_ocr",
                "api_model": self.api_model,
                "strip_markdown": self.strip_markdown,
                "output_empty": not bool(normalized),
                "raw_char_count": len(raw_text),
                "normalized_char_count": len(normalized),
            },
        )

    def close(self) -> None:
        self._client.close()
