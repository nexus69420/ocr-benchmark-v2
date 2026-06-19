"""Append-only run ledger (JSONL)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EventType = Literal[
    "run_created",
    "run_started",
    "unit_skipped",
    "unit_started",
    "unit_completed",
    "unit_failed",
    "run_finished",
]


class RunLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: EventType, **fields: Any) -> dict[str, Any]:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def unit_status(self, model: str, golden_id: str) -> str | None:
        """Latest status for a work unit from ledger events."""
        status = None
        for row in self.read_all():
            if row.get("model") != model or row.get("golden_id") != golden_id:
                continue
            ev = row.get("event")
            if ev == "unit_completed":
                status = "ok"
            elif ev == "unit_failed":
                status = "error"
            elif ev == "unit_skipped":
                status = "skipped"
            elif ev == "unit_started":
                status = "running"
        return status
