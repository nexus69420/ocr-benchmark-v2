"""Runner result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OcrRunResult:
    raw_text: str
    normalized_text: str
    latency_sec: float
    raw_payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
