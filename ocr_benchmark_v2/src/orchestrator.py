"""Resumable run orchestration."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config_loader import all_model_ids, load_models_config
from .dataset import GoldenPage, load_pages
from .paths import REPO_ROOT
from .run_context import RunContext
from .runners import close_runners, get_runner
from .runners.retry import is_quota_limit, quota_limit_message
from .unit_io import write_unit_error, write_unit_success


@dataclass
class WorkUnit:
    model: str
    page: GoldenPage
    golden_id: str
    status: str  # pending | skipped | ok | error | running
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "golden_id": self.golden_id,
            "doc_stem": self.page.doc_stem,
            "page": self.page.page,
            "status": self.status,
            "reason": self.reason,
        }


class RunOrchestrator:
    def __init__(self, ctx: RunContext, *, models: list[str] | None = None, force: bool = False) -> None:
        self.ctx = ctx
        self.force = force
        cfg = load_models_config()
        self.models = models or all_model_ids(cfg)

    def work_units(self) -> list[WorkUnit]:
        pages = load_pages()
        units: list[WorkUnit] = []
        for model in self.models:
            for page in pages:
                status, reason = self._unit_state(model, page.golden_id)
                units.append(
                    WorkUnit(
                        model=model,
                        page=page,
                        golden_id=page.golden_id,
                        status=status,
                        reason=reason,
                    )
                )
        return units

    def _unit_state(self, model: str, golden_id: str) -> tuple[str, str | None]:
        if not self.force and self.ctx.is_unit_complete(model, golden_id):
            return "skipped", "output_exists"
        if not self.force:
            ledger_status = self.ctx.ledger.unit_status(model, golden_id)
            if ledger_status == "ok":
                return "skipped", "ledger_ok"
        return "pending", None

    def plan(self) -> dict[str, Any]:
        units = self.work_units()
        return {
            "run_id": self.ctx.run_id,
            "run_root": str(self.ctx.root),
            "total_units": len(units),
            "pending": sum(1 for u in units if u.status == "pending"),
            "skipped": sum(1 for u in units if u.status == "skipped"),
            "units": [u.to_dict() for u in units],
        }

    def log_plan(self) -> dict[str, Any]:
        self.ctx.ledger.append("run_started", run_id=self.ctx.run_id, force=self.force)
        plan = self.plan()
        for unit in plan["units"]:
            if unit["status"] == "skipped":
                self.ctx.ledger.append(
                    "unit_skipped",
                    model=unit["model"],
                    golden_id=unit["golden_id"],
                    reason=unit.get("reason"),
                )
        return plan

    def execute(self) -> dict[str, Any]:
        load_dotenv(REPO_ROOT / ".env")
        self.ctx.ensure_dirs()
        plan = self.log_plan()
        pending = [u for u in self.work_units() if u.status == "pending"]

        if not pending:
            self.ctx.ledger.append("run_finished", run_id=self.ctx.run_id, status="noop")
            return plan

        by_model: dict[str, list[WorkUnit]] = defaultdict(list)
        for unit in pending:
            by_model[unit.model].append(unit)

        completed = 0
        failed = 0

        try:
            for model, units in sorted(by_model.items()):
                print(f"\n=== {model} ({len(units)} pages) ===", flush=True)
                runner = get_runner(model)
                for unit in units:
                    print(
                        f"OCR {unit.golden_id} {unit.page.doc_stem}/{unit.page.page} ...",
                        flush=True,
                    )
                    self.ctx.ledger.append(
                        "unit_started",
                        model=unit.model,
                        golden_id=unit.golden_id,
                    )
                    try:
                        result = runner.run_page(unit.page.image_path, unit.page)
                        write_unit_success(self.ctx, unit.model, unit.golden_id, unit.page, result)
                        self.ctx.ledger.append(
                            "unit_completed",
                            model=unit.model,
                            golden_id=unit.golden_id,
                            latency_sec=result.latency_sec,
                            chars=result.meta.get("normalized_char_count"),
                        )
                        completed += 1
                        print(
                            f"  ok {result.latency_sec:.1f}s | "
                            f"chars={result.meta.get('normalized_char_count')} "
                            f"truncated={result.meta.get('output_truncated', False)}",
                            flush=True,
                        )
                    except Exception as e:
                        failed += 1
                        err = f"{type(e).__name__}: {e}"
                        if is_quota_limit(e):
                            print("\n*** " + quota_limit_message(e) + " ***\n", flush=True)
                        write_unit_error(self.ctx, unit.model, unit.golden_id, unit.page, err)
                        self.ctx.ledger.append(
                            "unit_failed",
                            model=unit.model,
                            golden_id=unit.golden_id,
                            error=err,
                        )
                        print(f"  FAILED: {err}", flush=True)
                        if os.getenv("OCR_BENCHMARK_STRICT", "").lower() in ("1", "true", "yes"):
                            raise
        finally:
            close_runners()

        status = "ok" if failed == 0 else ("partial" if completed else "failed")
        self.ctx.ledger.append(
            "run_finished",
            run_id=self.ctx.run_id,
            status=status,
            completed=completed,
            failed=failed,
        )
        final = self.plan()
        final["completed"] = completed
        final["failed"] = failed
        final["status"] = status
        return final
