from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Optional

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

def _best_checkpoint_from_run_summary(run_dir: Path) -> Optional[Path]:
    rs = run_dir / "run_summary.json"
    if not rs.exists():
        return None
    try:
        data = json.loads(rs.read_text(encoding="utf-8"))
        best = data.get("best_model_checkpoint")
        if isinstance(best, str) and best:
            p = Path(best).expanduser()
            if not p.is_absolute():
                p = (run_dir / p).resolve()
            if p.exists():
                return p
    except Exception:
        return None
    return None

def resolve_checkpoint_path(path_or_repo: str) -> str:
    raw = str(path_or_repo)
    expanded = os.path.expanduser(raw)
    p = Path(expanded)
    if p.exists():
        p = p.resolve()
        if (p / "config.json").exists() and ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
            return str(p)
        if p.is_dir():
            best = _best_checkpoint_from_run_summary(p)
            if best is not None:
                return str(best)
            ckpts = []
            for child in p.iterdir():
                if child.is_dir():
                    m = _CKPT_RE.match(child.name)
                    if m:
                        ckpts.append((int(m.group(1)), child))
            if ckpts:
                ckpts.sort(key=lambda x: x[0])
                return str(ckpts[-1][1].resolve())
        return str(p)
    raise FileNotFoundError(f"Checkpoint path not found locally: {raw} (expanded: {expanded})")
