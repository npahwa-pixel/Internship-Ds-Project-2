# src/infer_json.py
from __future__ import annotations

import argparse
import json
import re
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Regex tokenizer that returns tokens + char offsets from original text
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, offsets = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def bio_to_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
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


def _load_tokenizer(model_dir: str):
    # ✅ Do NOT force slow->fast conversion. That’s what triggers the byte-fallback warning.
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)


def predict_word_tags(tokens: List[str], tokenizer, model, id2label: Dict[int, str], device) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()

    # word_ids() requires FAST tokenizer (we kept use_fast=True)
    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=256).word_ids()
    tags = ["O"] * len(tokens)

    seen = set()
    for i, w in enumerate(word_ids):
        if w is None or w in seen:
            continue
        seen.add(w)
        tags[w] = id2label[int(pred_ids[i])]
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Path to trained model dir")
    ap.add_argument("--text", type=str, default=None, help="If not provided, reads from stdin")
    args = ap.parse_args()

    text = args.text
    if text is None:
        import sys
        text = sys.stdin.read().strip()

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        print(json.dumps({"entities": []}, ensure_ascii=False))
        return

    tokenizer = _load_tokenizer(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    tags = predict_word_tags(tokens, tokenizer, model, id2label, device)
    spans = bio_to_spans(tags)

    entities: List[Dict[str, Any]] = []
    for typ, s, e in spans:
        start = offsets[s][0]
        end = offsets[e - 1][1]
        entities.append({"type": typ, "text": text[start:end], "start": start, "end": end})

    print(json.dumps({"entities": entities}, ensure_ascii=False))


if __name__ == "__main__":
    main()
