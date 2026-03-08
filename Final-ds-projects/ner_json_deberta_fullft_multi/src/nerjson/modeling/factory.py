from __future__ import annotations

import os
from transformers import AutoModelForTokenClassification, AutoTokenizer

from nerjson.config.labels import ID2LABEL, LABEL2ID, UNIFIED_TAGS
from nerjson.modeling.memory import enable_gradient_checkpointing
from nerjson.modeling.resolve import resolve_checkpoint_path


def load_tokenizer(model_name_or_path: str):
    resolved = resolve_checkpoint_path(model_name_or_path) if os.path.exists(os.path.expanduser(model_name_or_path)) else model_name_or_path
    # If resolved is a local folder, prefer local files only to avoid accidental hub calls
    local_only = os.path.isdir(os.path.expanduser(resolved))
    return AutoTokenizer.from_pretrained(resolved, use_fast=True, local_files_only=local_only)


def load_model_tokencls(model_name_or_path: str, attn_eager: bool = True):
    resolved = resolve_checkpoint_path(model_name_or_path) if os.path.exists(os.path.expanduser(model_name_or_path)) else model_name_or_path

    kwargs = dict(
        num_labels=len(UNIFIED_TAGS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=os.path.isdir(os.path.expanduser(resolved)),
    )
    if attn_eager:
        # DeBERTa v3 often needs eager attention on some backends
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForTokenClassification.from_pretrained(resolved, **kwargs)

    # Disable KV cache (saves memory)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    enable_gradient_checkpointing(model)
    return model
