"""Retry and rate-limit helpers for API runners."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "503", "unavailable", "timeout", "rate"))


def is_quota_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "quota",
            "resource_exhausted",
            "rate limit",
            "rate_limit",
            "exceeded your current quota",
            "too many requests",
        )
    )


def quota_limit_message(exc: BaseException) -> str:
    return (
        "GEMINI/API QUOTA OR RATE LIMIT REACHED — wait for reset, enable billing, "
        "or use a different Google Cloud project. Changing the API key alone usually does not help.\n"
        f"Detail: {exc}"
    )


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    backoff_base_sec: float,
    backoff_max_sec: float,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    last_err: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not is_retryable(e):
                raise
            wait = min(backoff_max_sec, backoff_base_sec ** attempt)
            if on_retry:
                on_retry(attempt, e, wait)
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


class RateLimiter:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval_sec <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

    def mark(self) -> None:
        self._last_call = time.monotonic()
