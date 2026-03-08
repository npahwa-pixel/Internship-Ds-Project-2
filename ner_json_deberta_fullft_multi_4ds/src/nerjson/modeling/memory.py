from __future__ import annotations
import gc
from typing import Tuple
import torch
from transformers import TrainerCallback

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class MpsEmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 20):
        self.every_n_steps = every_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and (state.global_step % self.every_n_steps == 0):
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            gc.collect()
        return control

def enable_gradient_checkpointing(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

def freeze_bottom_layers(model, n_freeze: int):
    if n_freeze <= 0:
        return
    base = None
    if hasattr(model, "deberta"):
        base = model.deberta
    elif hasattr(model, "deberta_v2"):
        base = model.deberta_v2
    if base is None:
        print("WARN: Could not locate base DeBERTa module; skipping freezing."); return
    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
    layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        layers = base.encoder.layer
    if layers is None:
        print("WARN: Could not locate encoder layers; skipping freezing."); return
    n_freeze = min(n_freeze, len(layers))
    for layer in layers[:n_freeze]:
        for p in layer.parameters():
            p.requires_grad = False
    print(f"Froze embeddings + bottom {n_freeze}/{len(layers)} layers.")

def count_trainable_params(model) -> Tuple[int, int]:
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
