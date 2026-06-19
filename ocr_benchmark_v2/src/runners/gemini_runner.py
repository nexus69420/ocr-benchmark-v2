"""Gemini VLM OCR with continuation on truncation."""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from ..config_loader import load_models_config, load_rate_limits
from ..dataset import GoldenPage
from ..normalize import looks_truncated, normalize_model_output
from ..prompts import CONTINUATION_SUFFIX, load_ocr_prompt
from .retry import RateLimiter, call_with_retry
from .types import OcrRunResult


def _compress_image(path: Path, max_side: int = 2048) -> tuple[bytes, str]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _response_to_dict(response: Any) -> dict[str, Any]:
    for attr in ("model_dump", "to_dict"):
        fn = getattr(response, attr, None)
        if callable(fn):
            try:
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                try:
                    return fn()
                except Exception:
                    pass
    try:
        import json

        return json.loads(response.json())  # type: ignore[attr-defined]
    except Exception:
        return {"text": getattr(response, "text", None)}


def _finish_reason_max_tokens(response: Any) -> bool:
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return False
        fr = getattr(candidates[0], "finish_reason", None)
        return fr is not None and "MAX_TOKENS" in str(fr).upper()
    except Exception:
        return False


def _needs_continuation(text: str, response: Any) -> bool:
    return looks_truncated(text) or _finish_reason_max_tokens(response)


class GeminiRunner:
    model_id = "gemini-ocr"

    def __init__(self) -> None:
        cfg = load_models_config()["models"][self.model_id]
        limits = load_rate_limits().get(self.model_id, {})
        self.api_model = os.getenv(cfg["api_model_env"], cfg["default_api_model"])
        self.max_output_tokens = int(cfg.get("max_output_tokens", 16384))
        self.max_continuation_passes = int(cfg.get("max_continuation_passes", 3))
        self.continuation_enabled = bool(cfg.get("continuation_on_truncation", True))
        self._limiter = RateLimiter(float(limits.get("min_interval_sec", 0)))
        self._max_retries = int(limits.get("max_retries", 5))
        self._backoff_base = float(limits.get("backoff_base_sec", 2))
        self._backoff_max = float(limits.get("backoff_max_sec", 60))
        self._client = None
        self._prompt = load_ocr_prompt()

    def _client_instance(self):
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    def _generate(self, contents: list[Any]) -> tuple[str, Any, float]:
        from google.genai import types

        client = self._client_instance()
        t0 = time.perf_counter()

        def _call():
            return client.models.generate_content(
                model=self.api_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                ),
            )

        response = call_with_retry(
            _call,
            max_retries=self._max_retries,
            backoff_base_sec=self._backoff_base,
            backoff_max_sec=self._backoff_max,
            on_retry=lambda attempt, err, wait: print(
                f"  gemini retry {attempt}/{self._max_retries} in {wait:.0f}s ({err})",
                flush=True,
            ),
        )
        elapsed = time.perf_counter() - t0
        text = (getattr(response, "text", None) or "").strip()
        return text, response, elapsed

    def run_page(self, image_path: Path, page: GoldenPage) -> OcrRunResult:
        from google.genai import types

        self._limiter.wait()
        image_bytes, mime = _compress_image(image_path)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)

        t0 = time.perf_counter()
        text, response, _ = self._generate([image_part, self._prompt])
        raw_calls: list[dict[str, Any]] = [
            {"pass": 1, "kind": "initial", "response": _response_to_dict(response), "text_chars": len(text)}
        ]
        passes = 1
        truncated = _needs_continuation(text, response)

        while (
            self.continuation_enabled
            and truncated
            and passes < self.max_continuation_passes
        ):
            passes += 1
            tail = text[-3000:] if len(text) > 3000 else text
            cont_prompt = (
                f"Text extracted so far (do not repeat):\n{tail}\n\n{CONTINUATION_SUFFIX}"
            )
            cont_text, cont_response, _ = self._generate([image_part, self._prompt, cont_prompt])
            raw_calls.append(
                {
                    "pass": passes,
                    "kind": "continuation",
                    "response": _response_to_dict(cont_response),
                    "text_chars": len(cont_text),
                }
            )
            if cont_text:
                if text and not text.endswith("\n") and not cont_text.startswith("\n"):
                    text += "\n"
                text += cont_text
            truncated = _needs_continuation(cont_text, cont_response) if cont_text else False

        self._limiter.mark()
        latency = time.perf_counter() - t0
        normalized = normalize_model_output(text, strip_markdown=False)
        still_truncated = looks_truncated(text) or (
            truncated and passes >= self.max_continuation_passes
        )

        return OcrRunResult(
            raw_text=text,
            normalized_text=normalized,
            latency_sec=round(latency, 3),
            raw_payload={
                "api_model": self.api_model,
                "prompt_sha256_prefix": self._prompt[:80],
                "continuation_passes": passes,
                "calls": raw_calls,
            },
            meta={
                "runner_type": "vlm_prompt",
                "api_model": self.api_model,
                "continuation_passes": passes,
                "output_truncated": still_truncated,
                "output_empty": not bool(normalized),
                "raw_char_count": len(text),
                "normalized_char_count": len(normalized),
            },
        )

    def close(self) -> None:
        self._client = None
