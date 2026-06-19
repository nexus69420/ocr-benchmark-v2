"""Gemini image-grounded OCR judge for v2 benchmark runs."""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from ..judge.gemini_judge_prompt import JUDGE_RESPONSE_SCHEMA, JUDGE_SYSTEM_PROMPT

from .config_loader import candidate_model_ids
from .dataset import load_pages
from .llm_client import get_llm_client, judge_model_id, resolve_provider
from .paths import REPO_ROOT
from .run_context import RunContext

CANDIDATE_MODELS = candidate_model_ids()


def get_client():
    return get_llm_client()


def compress_image(path: Path, max_side: int = 2048) -> tuple[bytes, str]:
    from io import BytesIO

    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue(), "image/jpeg"


def parse_judgment_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validate_judgment(data: dict) -> dict:
    missing = [k for k in JUDGE_RESPONSE_SCHEMA["required"] if k not in data]
    if missing:
        raise ValueError(f"Judgment missing keys: {missing}")
    return data


def _run_judge_google(
    client,
    judge_model: str,
    image_path: Path,
    user_block: str,
    retries: int,
) -> tuple[dict, float, str]:
    from google.genai import types

    image_bytes, mime = compress_image(image_path)
    t0 = time.perf_counter()
    last_err: Exception | None = None
    compact_hint = (
        "\n\nKeep each error array to at most 5 concise examples (max 80 characters each)."
    )

    for attempt in range(1, retries + 1):
        prompt = user_block + (compact_hint if attempt > 1 else "")
        try:
            response = client.models.generate_content(
                model=judge_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=16384,
                    response_mime_type="application/json",
                    response_schema=JUDGE_RESPONSE_SCHEMA,
                ),
            )
            elapsed = time.perf_counter() - t0
            raw = (response.text or "").strip()
            return _validate_judgment(parse_judgment_json(raw)), elapsed, raw
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"    json retry {attempt}/{retries} ({e})", flush=True)
            time.sleep(min(30, 2**attempt))
            continue
        except Exception as e:
            last_err = e
            if _is_retryable(e):
                wait = min(60, 2**attempt)
                print(f"    retry {attempt}/{retries} in {wait}s ({e})", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_err  # type: ignore[misc]


def _run_judge_openrouter(
    client,
    judge_model: str,
    image_path: Path,
    user_block: str,
    retries: int,
) -> tuple[dict, float, str]:
    import base64

    image_bytes, mime = compress_image(image_path)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    schema_hint = (
        "\n\nRespond with JSON only matching this schema:\n"
        f"{json.dumps(JUDGE_RESPONSE_SCHEMA, ensure_ascii=False)}"
    )
    t0 = time.perf_counter()
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        prompt = user_block + schema_hint
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=16384,
                response_format={"type": "json_object"},
            )
            elapsed = time.perf_counter() - t0
            raw = (response.choices[0].message.content or "").strip()
            return _validate_judgment(parse_judgment_json(raw)), elapsed, raw
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"    json retry {attempt}/{retries} ({e})", flush=True)
            time.sleep(min(30, 2**attempt))
            continue
        except Exception as e:
            last_err = e
            if _is_retryable(e):
                wait = min(60, 2**attempt)
                print(f"    retry {attempt}/{retries} in {wait}s ({e})", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last_err  # type: ignore[misc]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "503", "unavailable", "resource", "quota", "rate"))


def run_judge_call(
    client,
    judge_model: str,
    image_path: Path,
    ocr_plain_text: str,
    retries: int = 5,
) -> tuple[dict, float, str]:
    user_block = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        "## OCR output to evaluate (plain text)\n\n"
        f"{ocr_plain_text}"
    )
    if resolve_provider() == "openrouter":
        return _run_judge_openrouter(client, judge_model, image_path, user_block, retries)
    return _run_judge_google(client, judge_model, image_path, user_block, retries)


def judgment_path(ctx: RunContext, golden_id: str, model: str) -> Path:
    return ctx.root / "judge" / model / f"{golden_id}.json"


def run_judge(
    ctx: RunContext,
    *,
    models: list[str] | None = None,
    judge_model: str | None = None,
    force: bool = False,
    retries: int = 5,
) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    models = models or CANDIDATE_MODELS
    provider = resolve_provider()
    judge_model = judge_model or judge_model_id(provider)
    client = get_client()
    rpm_sleep = float(os.getenv("GEMINI_JUDGE_RPM_SLEEP", "2" if provider == "openrouter" else "13"))
    print(f"judge provider={provider} model={judge_model}", flush=True)

    completed = skipped = failed = 0
    pages = load_pages()

    for page in pages:
        for model in models:
            out_path = judgment_path(ctx, page.golden_id, model)
            ocr_path = ctx.output_txt(model, page.golden_id)

            if not force and out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    if existing.get("meta", {}).get("status") == "ok":
                        print(f"skip {page.golden_id}/{model}", flush=True)
                        skipped += 1
                        continue
                except json.JSONDecodeError:
                    pass

            if not ocr_path.exists():
                print(f"missing OCR {ocr_path}", flush=True)
                failed += 1
                continue
            if not page.image_path.exists():
                print(f"missing image {page.image_path}", flush=True)
                failed += 1
                continue

            ocr_plain = ocr_path.read_text(encoding="utf-8").strip()
            print(f"judge {page.golden_id} {page.doc_stem}/{page.page} {model} ...", flush=True)

            try:
                judgment, elapsed, raw = run_judge_call(
                    client, judge_model, page.image_path, ocr_plain, retries=retries
                )
                ref_meta_path = ctx.reference_dir / f"{page.golden_id}.meta.json"
                ref_truncated = False
                if ref_meta_path.exists():
                    ref_truncated = bool(
                        json.loads(ref_meta_path.read_text(encoding="utf-8")).get(
                            "reference_truncated"
                        )
                    )

                payload = {
                    "meta": {
                        "run_id": ctx.run_id,
                        "golden_id": page.golden_id,
                        "doc_stem": page.doc_stem,
                        "page": page.page,
                        "candidate_model": model,
                        "judge_model": judge_model,
                        "status": "ok",
                        "latency_sec": round(elapsed, 2),
                        "image_path": str(page.image_path),
                        "ocr_path": str(ocr_path),
                        "ocr_plain_chars": len(ocr_plain),
                        "reference_truncated": ref_truncated,
                        "judged_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "judgment": judgment,
                    "raw_response": raw,
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"  {elapsed:.1f}s | score={judgment.get('overall_score')} "
                    f"quality={judgment.get('quality')}",
                    flush=True,
                )
                completed += 1
            except Exception as e:
                failed += 1
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "meta": {
                                "golden_id": page.golden_id,
                                "candidate_model": model,
                                "status": "error",
                                "error": str(e),
                            },
                            "judgment": None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  ERROR: {e}", flush=True)

            if rpm_sleep > 0:
                time.sleep(rpm_sleep)

    return {"completed": completed, "skipped": skipped, "failed": failed}


def export_judge_csv(ctx: RunContext, *, models: list[str] | None = None) -> dict:
    models = models or CANDIDATE_MODELS
    pages = load_pages()
    rows: list[dict] = []

    for page in pages:
        for model in models:
            jpath = judgment_path(ctx, page.golden_id, model)
            row: dict = {
                "run_id": ctx.run_id,
                "golden_id": page.golden_id,
                "model": model,
                "doc_stem": page.doc_stem,
                "page": page.page,
                "judge_status": None,
                "overall_score": None,
                "quality": None,
                "estimated_character_accuracy": None,
                "verdict": None,
                "latency_sec": None,
                "ocr_chars": None,
            }
            if jpath.exists():
                record = json.loads(jpath.read_text(encoding="utf-8"))
                m = record.get("meta", {})
                j = record.get("judgment") or {}
                ocr_path = ctx.output_txt(model, page.golden_id)
                row.update(
                    {
                        "judge_status": m.get("status"),
                        "overall_score": j.get("overall_score"),
                        "quality": j.get("quality"),
                        "estimated_character_accuracy": j.get("estimated_character_accuracy"),
                        "verdict": j.get("verdict"),
                        "latency_sec": m.get("latency_sec"),
                        "ocr_chars": len(ocr_path.read_text(encoding="utf-8"))
                        if ocr_path.exists()
                        else None,
                    }
                )
            rows.append(row)

    judge_dir = ctx.root / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    csv_path = judge_dir / "judge_scores.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rollup: dict[str, dict] = {}
    for model in models:
        scored = [
            r
            for r in rows
            if r["model"] == model and r["judge_status"] == "ok" and r["overall_score"] is not None
        ]
        if not scored:
            rollup[model] = {"pages_judged": 0}
        else:
            rollup[model] = {
                "pages_judged": len(scored),
                "avg_overall_score": round(
                    sum(float(x["overall_score"]) for x in scored) / len(scored), 2
                ),
                "avg_character_accuracy": round(
                    sum(float(x["estimated_character_accuracy"]) for x in scored) / len(scored),
                    2,
                ),
            }

    rollup_path = judge_dir / "judge_rollup.json"
    rollup_path.write_text(json.dumps(rollup, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "rollup": rollup, "rollup_path": str(rollup_path)}
