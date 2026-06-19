"""Versioned benchmark run directories and config snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import all_model_ids, load_models_config, load_rate_limits
from .dataset import load_manifest, manifest_sha256
from .ledger import RunLedger
from .paths import DEFAULT_PROMPT_FILE, REPO_ROOT, RUNS_ROOT


def prompt_sha256(path: Path | None = None) -> str:
    path = path or DEFAULT_PROMPT_FILE
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class RunContext:
    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = (root or RUNS_ROOT) / run_id
        self.ledger = RunLedger(self.root / "ledger.jsonl")
        self.snapshot_path = self.root / "config.snapshot.json"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def scores_dir(self) -> Path:
        return self.root / "scores"

    def output_txt(self, model: str, golden_id: str) -> Path:
        return self.outputs_dir / model / f"{golden_id}.txt"

    def output_meta(self, model: str, golden_id: str) -> Path:
        return self.outputs_dir / model / f"{golden_id}.meta.json"

    def raw_json(self, model: str, golden_id: str) -> Path:
        return self.raw_dir / model / f"{golden_id}.json"

    def reference_txt(self, golden_id: str) -> Path:
        return self.reference_dir / f"{golden_id}.txt"

    def ensure_dirs(self) -> None:
        for d in (self.outputs_dir, self.raw_dir, self.reference_dir, self.scores_dir):
            d.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest = load_manifest()
        try:
            prompt_rel = str(DEFAULT_PROMPT_FILE.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            prompt_rel = str(DEFAULT_PROMPT_FILE)
        snapshot = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "framework_version": "2.0.0",
            "golden_set_id": manifest.get("golden_set_id"),
            "golden_set_version": manifest.get("version"),
            "golden_manifest_sha256": manifest_sha256(),
            "prompt_file": prompt_rel,
            "prompt_sha256": prompt_sha256(),
            "normalization": {
                "unicode": "NFC",
                "preserve_newlines": True,
                "collapse": "horizontal_whitespace_only",
                "mistral_strip_markdown": True,
            },
            "models": load_models_config(),
            "rate_limits": load_rate_limits(),
            "model_ids": all_model_ids(),
            **(extra or {}),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot

    @classmethod
    def create(cls, run_id: str | None = None) -> RunContext:
        rid = run_id or new_run_id()
        ctx = cls(rid)
        ctx.ensure_dirs()
        ctx.write_snapshot()
        ctx.ledger.append("run_created", run_id=rid)
        return ctx

    @classmethod
    def load(cls, run_id: str) -> RunContext:
        ctx = cls(run_id)
        if not ctx.root.exists():
            raise FileNotFoundError(f"Run not found: {ctx.root}")
        return ctx

    def load_meta(self, model: str, golden_id: str) -> dict[str, Any] | None:
        p = self.output_meta(model, golden_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def is_unit_complete(self, model: str, golden_id: str) -> bool:
        txt = self.output_txt(model, golden_id)
        meta = self.output_meta(model, golden_id)
        if not txt.exists() or not meta.exists():
            return False
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return data.get("status") == "ok"
        except json.JSONDecodeError:
            return False
