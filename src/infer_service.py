# src/infer_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import html
import torch

from .model_loader import LoadedModel
from .text_tokenize import tokenize_with_offsets


@dataclass
class InferenceResult:
    entities: List[Dict[str, Any]]
    tokens: List[str]
    tags: List[str]


# Backward-compatible alias (in case other files used InferResult)
InferResult = InferenceResult


def _bio_to_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t == "O":
            i += 1
            continue
        if t.startswith("B-"):
            typ = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append((typ, i, j))
            i = j
        elif t.startswith("I-"):
            typ = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append((typ, i, j))
            i = j
        else:
            i += 1
    return spans


def _predict_word_tags(tokens: List[str], tokenizer, model, id2label: Dict[int, str], device) -> List[str]:
    # EXACT match to your original infer_json.py
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()

    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=256).word_ids()
    tags = ["O"] * len(tokens)

    seen = set()
    for i, w in enumerate(word_ids):
        if w is None or w in seen:
            continue
        seen.add(w)
        tags[w] = id2label[int(pred_ids[i])]
    return tags


def infer_text(text: str, lm: LoadedModel) -> InferenceResult:
    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return InferenceResult(entities=[], tokens=[], tags=[])

    tags = _predict_word_tags(tokens, lm.tokenizer, lm.model, lm.id2label, lm.device)
    spans = _bio_to_spans(tags)

    entities: List[Dict[str, Any]] = []
    for typ, s, e in spans:
        start = offsets[s][0]
        end = offsets[e - 1][1]
        entities.append({"type": typ, "text": text[start:end], "start": start, "end": end})

    return InferenceResult(entities=entities, tokens=tokens, tags=tags)


def render_highlight_html(text: str, entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return f"<div style='white-space:pre-wrap'>{html.escape(text)}</div>"

    ents = sorted(entities, key=lambda e: (int(e["start"]), int(e["end"])))
    out = []
    cur = 0
    for e in ents:
        s = int(e["start"])
        t = int(e["end"])
        if s < cur:
            continue
        out.append(html.escape(text[cur:s]))
        label = html.escape(str(e.get("type", "")))
        span_text = html.escape(text[s:t])
        out.append(
            f"<mark title='{label}' style='padding:0.05em 0.2em;border-radius:0.2em;'>"
            f"<b>{label}</b>: {span_text}</mark>"
        )
        cur = t
    out.append(html.escape(text[cur:]))
    return f"<div style='white-space:pre-wrap;line-height:1.6'>{''.join(out)}</div>"
