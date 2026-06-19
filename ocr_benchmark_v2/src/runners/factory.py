"""Runner factory with per-run caching."""

from __future__ import annotations

from typing import Any

_RUNNER_CACHE: dict[str, Any] = {}


def get_runner(model_id: str):
    if model_id in _RUNNER_CACHE:
        return _RUNNER_CACHE[model_id]

    if model_id == "gemini-ocr":
        from .gemini_runner import GeminiRunner

        runner = GeminiRunner()
    elif model_id == "mistral-ocr":
        from .mistral_runner import MistralRunner

        runner = MistralRunner()
    elif model_id == "chandra-2":
        from .chandra_runner import ChandraRunner

        runner = ChandraRunner()
    elif model_id == "qwen2.5-vl-7b":
        from .qwen_runner import QwenRunner

        runner = QwenRunner()
    elif model_id == "paddleocr-vl":
        from .paddle_runner import PaddleRunner

        runner = PaddleRunner()
    else:
        raise ValueError(f"Unknown model: {model_id}")

    _RUNNER_CACHE[model_id] = runner
    return runner


def close_runners() -> None:
    for runner in _RUNNER_CACHE.values():
        close = getattr(runner, "close", None)
        if callable(close):
            close()
    _RUNNER_CACHE.clear()
