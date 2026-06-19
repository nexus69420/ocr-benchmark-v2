"""Qwen2.5-VL local OCR runner."""

from __future__ import annotations

import time
from pathlib import Path

from ..config_loader import load_models_config
from ..dataset import GoldenPage
from ..hf_env import apply_hf_cache_env, from_pretrained_kwargs
from ..normalize import normalize_model_output
from ..prompts import load_ocr_prompt
from .types import OcrRunResult


class QwenRunner:
    model_id = "qwen2.5-vl-7b"

    def __init__(self) -> None:
        self._prompt = load_ocr_prompt()
        self._ctx = None
        cfg = load_models_config()["models"][self.model_id]
        self.hf_id = cfg.get("huggingface_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 8192))
        self._hf_env = apply_hf_cache_env()

    def _load(self):
        if self._ctx is not None:
            return self._ctx
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kw = from_pretrained_kwargs()
        processor = AutoProcessor.from_pretrained(self.hf_id, **kw)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_id,
            device_map="auto",
            torch_dtype=torch.float16,
            **kw,
        )
        self._ctx = (model, processor, Image, process_vision_info)
        return self._ctx

    def run_page(self, image_path: Path, page: GoldenPage) -> OcrRunResult:
        import torch

        model, processor, Image, process_vision_info = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_in],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        latency = time.perf_counter() - t0
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        raw_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        normalized = normalize_model_output(raw_text, strip_markdown=False)

        return OcrRunResult(
            raw_text=raw_text,
            normalized_text=normalized,
            latency_sec=round(latency, 3),
            raw_payload={
                "huggingface_id": self.hf_id,
                "max_new_tokens": self.max_new_tokens,
                "prompt_chars": len(self._prompt),
                "hf_cache": self._hf_env,
            },
            meta={
                "runner_type": "vlm_local",
                "huggingface_id": self.hf_id,
                "hf_hub_cache": self._hf_env.get("hf_hub_cache"),
                "output_empty": not bool(normalized),
                "raw_char_count": len(raw_text),
                "normalized_char_count": len(normalized),
            },
        )

    def close(self) -> None:
        self._ctx = None
