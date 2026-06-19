"""Load shared OCR prompts."""

from __future__ import annotations

from functools import lru_cache

from .paths import DEFAULT_PROMPT_FILE

CONTINUATION_SUFFIX = """
Continue extracting text from this document image exactly where you stopped.
Do not repeat any text you already output.
Output only the remaining text from the image, preserving the same fidelity rules.
Return only the extracted text with no commentary.
""".strip()


@lru_cache(maxsize=1)
def load_ocr_prompt() -> str:
    return DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
