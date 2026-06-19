"""LLM client factory: Google AI direct or OpenRouter (OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import Literal

Provider = Literal["google", "openrouter"]

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
GOOGLE_JUDGE_MODEL = "gemini-2.5-flash"
OPENROUTER_JUDGE_MODEL = "google/gemini-2.5-flash"


def resolve_provider() -> Provider:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("google", "openrouter"):
        return explicit  # type: ignore[return-value]
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "google"


def judge_model_id(provider: Provider | None = None) -> str:
    provider = provider or resolve_provider()
    env = os.getenv("GEMINI_JUDGE_MODEL")
    if env:
        return env
    return OPENROUTER_JUDGE_MODEL if provider == "openrouter" else GOOGLE_JUDGE_MODEL


def get_google_genai_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    from google import genai

    return genai.Client(api_key=api_key)


def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    from openai import OpenAI

    base = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE)
    return OpenAI(base_url=base, api_key=api_key)


def get_llm_client(provider: Provider | None = None):
    provider = provider or resolve_provider()
    if provider == "openrouter":
        return get_openrouter_client()
    return get_google_genai_client()
