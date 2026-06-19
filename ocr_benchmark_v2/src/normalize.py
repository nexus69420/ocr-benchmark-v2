"""Canonical text normalization for scoring."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\ufeff\u200b-\u200f\u202a-\u202e]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")


def strip_markdown_for_scoring(text: str) -> str:
    """Strip markdown/HTML before normalization (Mistral OCR output)."""
    text = re.sub(r"```.*?```", "\n", text, flags=re.DOTALL)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"^[-:| ]+$", "", text, flags=re.MULTILINE)
    return text


def normalize_for_scoring(text: str) -> str:
    """NFC, preserve newlines, collapse horizontal whitespace only."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    lines = []
    for line in text.splitlines():
        line = _HORIZONTAL_SPACE.sub(" ", line).strip()
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_model_output(text: str, *, strip_markdown: bool = False) -> str:
    if strip_markdown:
        text = strip_markdown_for_scoring(text)
    return normalize_for_scoring(text)


def looks_truncated(text: str) -> bool:
    """Heuristic: output ends mid-number or mid-word."""
    if not text:
        return False
    tail = text.rstrip()[-80:]
    if re.search(r"[0-9૦-૯.]+\s*$", tail):
        return True
    if re.search(r"\S+\s*$", tail) and not tail.endswith((".", "।", "!", "?", '"', "'")):
        # ends abruptly without sentence punctuation — weak signal
        if len(text) < 800:
            return True
    return False
