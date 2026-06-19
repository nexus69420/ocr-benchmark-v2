#!/usr/bin/env python3
"""CLI for OCR benchmark v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Allow `python -m ocr_benchmark_v2` from repo root
_V2 = Path(__file__).resolve().parent
if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))

from src.dataset import validate_dataset
from src.orchestrator import RunOrchestrator
from src.paths import RUNS_ROOT
from src.run_context import RunContext, new_run_id
from src.reference_builder import build_reference
from src.report import write_report
from src.scorer import score_run


def cmd_init(_: argparse.Namespace) -> None:
    script = _V2 / "golden_set" / "build_golden_set.py"
    subprocess.check_call([sys.executable, str(script)])


def cmd_validate(args: argparse.Namespace) -> None:
    report = validate_dataset()
    out = args.output
    if out:
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        sys.exit(1)


def cmd_run_create(args: argparse.Namespace) -> None:
    run_id = args.run_id or new_run_id()
    ctx = RunContext.create(run_id)
    print(json.dumps({"run_id": run_id, "run_root": str(ctx.root)}, indent=2))


def cmd_run_plan(args: argparse.Namespace) -> None:
    ctx = RunContext.load(args.run_id)
    orch = RunOrchestrator(ctx, models=args.model, force=args.force)
    plan = orch.log_plan()
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def cmd_run_execute(args: argparse.Namespace) -> None:
    ctx = RunContext.load(args.run_id)
    orch = RunOrchestrator(ctx, models=args.model, force=args.force)
    plan = orch.execute()
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if plan.get("failed"):
        sys.exit(1)


def cmd_run_list(_: argparse.Namespace) -> None:
    if not RUNS_ROOT.exists():
        print("No runs yet.")
        return
    for p in sorted(RUNS_ROOT.iterdir()):
        if p.is_dir():
            snap = p / "config.snapshot.json"
            print(p.name, "(snapshot ok)" if snap.exists() else "(incomplete)")


def cmd_run_reference(args: argparse.Namespace) -> None:
    ctx = RunContext.load(args.run_id)
    result = build_reference(ctx, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        sys.exit(1)


def cmd_run_score(args: argparse.Namespace) -> None:
    ctx = RunContext.load(args.run_id)
    result = score_run(ctx, models=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run_report(args: argparse.Namespace) -> None:
    ctx = RunContext.load(args.run_id)
    result = write_report(ctx)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run_judge(args: argparse.Namespace) -> None:
    from src.image_judge import export_judge_csv, run_judge

    ctx = RunContext.load(args.run_id)
    result = run_judge(ctx, models=args.model, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.export_csv:
        export = export_judge_csv(ctx, models=args.model)
        print(json.dumps(export, ensure_ascii=False, indent=2))
    if result.get("failed"):
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR benchmark v2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Build golden set from v1 (6 pages)")
    p_init.set_defaults(func=cmd_init)

    p_val = sub.add_parser("validate", help="Validate golden set images and hashes")
    p_val.add_argument("--output", type=Path, help="Write validation JSON to path")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="Versioned benchmark runs")
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)

    p_create = run_sub.add_parser("create", help="Create new run directory + snapshot")
    p_create.add_argument("--run-id", help="Run ID (default: UTC timestamp)")
    p_create.set_defaults(func=cmd_run_create)

    p_plan = run_sub.add_parser("plan", help="Show resumable work units")
    p_plan.add_argument("--run-id", required=True)
    p_plan.add_argument("--model", action="append", help="Limit to model(s)")
    p_plan.add_argument("--force", action="store_true", help="Ignore existing outputs")
    p_plan.set_defaults(func=cmd_run_plan)

    p_exec = run_sub.add_parser("execute", help="Run OCR on pending work units")
    p_exec.add_argument("--run-id", required=True)
    p_exec.add_argument("--model", action="append")
    p_exec.add_argument("--force", action="store_true")
    p_exec.set_defaults(func=cmd_run_execute)

    p_list = run_sub.add_parser("list", help="List run IDs")
    p_list.set_defaults(func=cmd_run_list)

    p_ref = run_sub.add_parser("reference", help="Build reference/ from Gemini outputs (M4)")
    p_ref.add_argument("--run-id", required=True)
    p_ref.add_argument("--force", action="store_true")
    p_ref.set_defaults(func=cmd_run_reference)

    p_score = run_sub.add_parser("score", help="Compute page_metrics.csv (M5)")
    p_score.add_argument("--run-id", required=True)
    p_score.add_argument("--model", action="append", help="Candidate model(s) only")
    p_score.set_defaults(func=cmd_run_score)

    p_report = run_sub.add_parser("report", help="Write scores/REPORT.md (M6)")
    p_report.add_argument("--run-id", required=True)
    p_report.set_defaults(func=cmd_run_report)

    p_judge = run_sub.add_parser("judge", help="Gemini image-grounded judge on candidate OCR")
    p_judge.add_argument("--run-id", required=True)
    p_judge.add_argument("--model", action="append", help="Candidate model(s) only")
    p_judge.add_argument("--force", action="store_true", help="Re-judge even if ok")
    p_judge.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Write judge/judge_scores.csv + rollup (default: on)",
    )
    p_judge.set_defaults(func=cmd_run_judge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
