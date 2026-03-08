from __future__ import annotations

import os
import re
import json
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
            # If stored as relative, anchor to run_dir
            if not p.is_absolute():
                p = (run_dir / p).resolve()
            if p.exists():
                return p
    except Exception:
        return None
    return None


def resolve_checkpoint_path(path_or_repo: str) -> str:
    """Resolve a user-supplied checkpoint path.

    Accepts:
    - a checkpoint directory that contains config.json / model.safetensors
    - a run directory that contains checkpoint-* subdirectories
    - an HF model id (if it is not a local path)

    If the path looks local but doesn't exist, raise a helpful error.
    """
    raw = str(path_or_repo)
    expanded = os.path.expanduser(raw)

    # If user gave a path that exists, resolve it.
    p = Path(expanded)
    if p.exists():
        p = p.resolve()

        # Case A: direct checkpoint folder
        if (p / "config.json").exists() and ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
            return str(p)

        # Case B: run dir containing checkpoints
        if p.is_dir():
            best = _best_checkpoint_from_run_summary(p)
            if best is not None:
                return str(best)

            ckpts = []
            for child in p.iterdir():
                if not child.is_dir():
                    continue
                m = _CKPT_RE.match(child.name)
                if m:
                    ckpts.append((int(m.group(1)), child))
            if ckpts:
                ckpts.sort(key=lambda x: x[0])
                return str(ckpts[-1][1].resolve())

        # Fall back to using directory as-is (might still work for tokenizer-only dirs)
        return str(p)

    # If it doesn't exist locally, transformers will treat it as repo id.
    # But many users *intended* a local relative path — so raise a clear message.
    raise FileNotFoundError(
        "Checkpoint path not found locally: "
        f"{raw} (expanded: {expanded})\n"
        "Fix: pass the correct relative path from your current working directory, or an absolute path.\n"
        "Examples:\n"
        "  --checkpoint ../deberta_fullft_multi/checkpoint-14319\n"
        "  --checkpoint /absolute/path/to/deberta_fullft_multi/checkpoint-14319\n"
        "Or pass the RUN DIR and the code will auto-pick the best/latest checkpoint:\n"
        "  --checkpoint ../deberta_fullft_multi\n"
    )
