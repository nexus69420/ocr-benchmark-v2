"""Load YAML config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")


def load_rate_limits() -> dict[str, Any]:
    return load_yaml("rate_limits.yaml")


def load_metrics_config() -> dict[str, Any]:
    return load_yaml("metrics.yaml")


def candidate_model_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_models_config()
    ref = reference_model_id(cfg)
    return [
        mid
        for mid, m in cfg.get("models", {}).items()
        if m.get("role") != "reference_only" and mid != ref
    ]


def all_model_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_models_config()
    return list(cfg.get("models", {}).keys())


def reference_model_id(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_models_config()
    return cfg.get("reference_model", "gemini-ocr")
