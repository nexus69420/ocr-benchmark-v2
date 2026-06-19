"""OCR model runners (M3)."""

from .factory import close_runners, get_runner
from .types import OcrRunResult

__all__ = ["OcrRunResult", "get_runner", "close_runners"]
