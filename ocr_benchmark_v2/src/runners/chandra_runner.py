"""Chandra OCR 2 local runner."""

from __future__ import annotations

import time
from pathlib import Path

from ..config_loader import load_models_config
from ..dataset import GoldenPage
from ..hf_env import apply_hf_cache_env, from_pretrained_kwargs
from ..normalize import normalize_model_output
from ..prompts import load_ocr_prompt
from .types import OcrRunResult


class ChandraRunner:
    model_id = "chandra-2"

    def __init__(self) -> None:
        self._ctx = None
        cfg = load_models_config()["models"][self.model_id]
        self.hf_id = cfg.get("huggingface_id", "datalab-to/chandra-ocr-2")
        self.max_output_tokens = int(cfg.get("max_output_tokens", 8192))
        self._prompt = load_ocr_prompt()
        self._hf_env = apply_hf_cache_env()

    def _load(self):
        if self._ctx is not None:
            return self._ctx
        import torch
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kw = from_pretrained_kwargs()
        model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            dtype=torch.bfloat16,
            device_map="auto",
            **kw,
        )
        model.eval()
        model.processor = AutoProcessor.from_pretrained(self.hf_id, **kw)
        model.processor.tokenizer.padding_side = "left"
        self._ctx = (model, Image, BatchInputItem, generate_hf)
        return self._ctx

    def run_page(self, image_path: Path, page: GoldenPage) -> OcrRunResult:
        model, Image, BatchInputItem, generate_hf = self._load()
        t0 = time.perf_counter()
        batch = [
            BatchInputItem(
                image=Image.open(image_path).convert("RGB"),
                prompt=self._prompt,
            )
        ]
        result = generate_hf(batch, model, max_output_tokens=self.max_output_tokens)[0]
        # Shared prompt requests plain text — do not run HTML layout markdown parser.
        raw_text = (result.raw or "").strip()
        latency = time.perf_counter() - t0
        normalized = normalize_model_output(raw_text, strip_markdown=False)

        return OcrRunResult(
            raw_text=raw_text,
            normalized_text=normalized,
            latency_sec=round(latency, 3),
            raw_payload={
                "huggingface_id": self.hf_id,
                "prompt_mode": "shared_ocr_extract_v1",
                "max_output_tokens": self.max_output_tokens,
                "prompt_chars": len(self._prompt),
                "hf_cache": self._hf_env,
                "raw_result": raw_text[:5000],
            },
            meta={
                "runner_type": "vlm_local",
                "huggingface_id": self.hf_id,
                "hf_hub_cache": self._hf_env.get("hf_hub_cache"),
                "uses_shared_prompt": True,
                "prompt_mode": "shared_ocr_extract_v1",
                "output_empty": not bool(normalized),
                "raw_char_count": len(raw_text),
                "normalized_char_count": len(normalized),
            },
        )

    def close(self) -> None:
        self._ctx = None
