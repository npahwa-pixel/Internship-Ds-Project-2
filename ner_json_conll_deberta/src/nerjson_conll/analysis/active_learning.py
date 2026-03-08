from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from .build_external_datasets import load_named_dataset
from .infer_service import InferenceResult, infer_text
from .model_loader import LoadedModel

PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "»", "”"}
PUNCT_NO_SPACE_AFTER  = {"(", "[", "{", "«", "“", "$"}

def reconstruct_text(tokens: List[str]) -> str:
    # Deterministic whitespace reconstruction (same style as preview_conll_json.py)
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            prefix = ""
        else:
            prev = tokens[i - 1]
            if tok in PUNCT_NO_SPACE_BEFORE:
                prefix = ""
            elif prev in PUNCT_NO_SPACE_AFTER:
                prefix = ""
            else:
                prefix = " "
        parts.append(prefix + tok)
    return "".join(parts)

@dataclass
class Candidate:
    text: str
    idx: int
    score: float
    tokens: List[str]

def uncertainty_score(res: InferenceResult) -> float:
    # Higher score => more uncertain.
    # Simple: 1 - mean confidence across tokens (ignore 0.0 which usually means truncation/empty)
    vals = [c for c in res.token_confs if c > 0.0]
    if not vals:
        return 1.0
    return float(1.0 - sum(vals) / len(vals))

def get_uncertain_sample(
    lm: LoadedModel,
    dataset: str = "conll",
    split: str = "validation",
    lang: str = "en",
    n_candidates: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    ds = load_named_dataset(dataset, lang=lang)[split]
    n = len(ds)
    if n == 0:
        raise ValueError("Dataset split is empty.")

    # Randomly pick indices to evaluate
    picks = [rng.randrange(0, n) for _ in range(min(n_candidates, n))]

    best: Candidate | None = None
    for idx in picks:
        row = ds[int(idx)]
        text = reconstruct_text(row["tokens"])
        res = infer_text(text, lm)
        score = uncertainty_score(res)
        if best is None or score > best.score:
            best = Candidate(text=text, idx=int(idx), score=score, tokens=row["tokens"])

    assert best is not None
    return {"text": best.text, "idx": best.idx, "score": best.score, "tokens": best.tokens}
