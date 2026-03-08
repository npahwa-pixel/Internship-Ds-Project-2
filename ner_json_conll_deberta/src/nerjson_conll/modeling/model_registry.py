from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Heuristics:
# - LoRA adapter dirs usually contain adapter_config.json
# - Full TokenClassification model dirs contain config.json + model.safetensors/pytorch_model.bin

@dataclass(frozen=True)
class ModelSpec:
    key: str               # stable identifier used by UI (e.g. "head:deberta_conll_head")
    display_name: str      # human readable
    kind: str              # "head" | "lora" | "other"
    path: Path             # directory on disk
    base_model: Optional[str] = None  # for LoRA adapters (read from adapter_config.json)

def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def discover_models(models_dir: str | Path = "models") -> List[ModelSpec]:
    root = Path(models_dir)
    if not root.exists():
        return []

    specs: List[ModelSpec] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        # ignore eval temp dirs
        if p.name.startswith("_tmp"):
            continue

        adapter_cfg = p / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = _safe_read_json(adapter_cfg) or {}
            base = cfg.get("base_model_name_or_path")
            key = f"lora:{p.name}"
            specs.append(ModelSpec(
                key=key,
                display_name=f"LoRA · {p.name}",
                kind="lora",
                path=p,
                base_model=base,
            ))
            continue

        # token classification model dir
        config_json = p / "config.json"
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        if config_json.exists() and has_weights:
            key = f"head:{p.name}"
            specs.append(ModelSpec(
                key=key,
                display_name=f"Model · {p.name}",
                kind="head",
                path=p,
                base_model=None,
            ))
            continue

        # Could be checkpoints inside stage dir; include them too (optional)
        # If user points dropdown at checkpoint-xxxx, we want it.
        # We'll scan one level deep for such dirs.
        if config_json.exists() and not has_weights:
            # might be a directory with checkpoints only; scan children
            for child in sorted(p.glob("checkpoint-*")):
                if not child.is_dir():
                    continue
                if (child / "config.json").exists() and ((child / "model.safetensors").exists() or (child / "pytorch_model.bin").exists()):
                    key = f"head:{p.name}/{child.name}"
                    specs.append(ModelSpec(
                        key=key,
                        display_name=f"Checkpoint · {p.name}/{child.name}",
                        kind="head",
                        path=child,
                        base_model=None,
                    ))

    return specs

def get_spec_by_key(specs: List[ModelSpec], key: str) -> ModelSpec:
    for s in specs:
        if s.key == key:
            return s
    raise KeyError(f"Unknown model key: {key}")
