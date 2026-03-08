from __future__ import annotations
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from nerjson.config.labels import ID2LABEL, LABEL2ID, UNIFIED_TAGS
from nerjson.modeling.memory import enable_gradient_checkpointing
from nerjson.modeling.resolve import resolve_checkpoint_path

def load_tokenizer(model_name_or_path: str):
    if os.path.exists(os.path.expanduser(model_name_or_path)):
        resolved = resolve_checkpoint_path(model_name_or_path)
        return AutoTokenizer.from_pretrained(resolved, use_fast=True, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def load_model_tokencls(model_name_or_path: str, attn_eager: bool = True):
    local_only = os.path.exists(os.path.expanduser(model_name_or_path))
    resolved = resolve_checkpoint_path(model_name_or_path) if local_only else model_name_or_path
    kwargs = dict(num_labels=len(UNIFIED_TAGS), id2label=ID2LABEL, label2id=LABEL2ID, local_files_only=local_only)
    if attn_eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForTokenClassification.from_pretrained(resolved, **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    enable_gradient_checkpointing(model)
    return model
