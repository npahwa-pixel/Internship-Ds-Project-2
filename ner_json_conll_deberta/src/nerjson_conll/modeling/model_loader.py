# src/model_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    id2label: Dict[int, str]
    label2id: Dict[str, int]


# Backward compatible alias
LoadedNERModel = LoadedModel


def _resolve_device(device_str: str) -> torch.device:
    d = (device_str or "cpu").strip().lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_head_tokenizer(model_dir: str):
    # EXACTLY match your original working inference:
    # AutoTokenizer(... use_fast=True, from_slow=True)
    return AutoTokenizer.from_pretrained(model_dir, use_fast=True, from_slow=True)


def _load_head_model(model_dir: str):
    # Force local weights only (prevents accidental remote fallback).
    try:
        return AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    except TypeError:
        # Some older transformers may not support local_files_only for this call path.
        return AutoModelForTokenClassification.from_pretrained(model_dir)


def load_any_model(kind: str, path: Union[str, Path], device_str: str = "cpu") -> LoadedModel:
    """
    kind: "head" or "lora"
    path:
      - head: folder with config.json + model.safetensors (or pytorch_model.bin)
      - lora: folder with adapter_config.json + adapter_model.safetensors
    """
    device = _resolve_device(device_str)
    k = (kind or "head").strip().lower()
    p = Path(path).expanduser().resolve()

    if k == "lora":
        try:
            from peft import PeftConfig, PeftModel  # type: ignore
        except Exception as e:
            raise RuntimeError("peft is required to load LoRA adapters. Install: pip install peft") from e

        peft_cfg = PeftConfig.from_pretrained(str(p))
        base = peft_cfg.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, from_slow=True)
        base_model = AutoModelForTokenClassification.from_pretrained(base)
        model = PeftModel.from_pretrained(base_model, str(p))
    else:
        # HEAD model (final dir or checkpoint dir)
        tokenizer = _load_head_tokenizer(str(p))
        model = _load_head_model(str(p))

    model.to(device)
    model.eval()

    id2label_raw = getattr(model.config, "id2label", {}) or {}
    label2id_raw = getattr(model.config, "label2id", {}) or {}

    id2label: Dict[int, str] = {int(kk): str(vv) for kk, vv in id2label_raw.items()}
    label2id: Dict[str, int] = {str(kk): int(vv) for kk, vv in label2id_raw.items()}

    if not label2id and id2label:
        label2id = {v: k for k, v in id2label.items()}
    if not id2label and label2id:
        id2label = {v: k for k, v in label2id.items()}

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        id2label=id2label,
        label2id=label2id,
    )
