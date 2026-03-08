# src/preview_conll_json.py
from __future__ import annotations

import argparse
import json
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from nerjson_conll.data.conll import load_conll

PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "»", "”"}
PUNCT_NO_SPACE_AFTER  = {"(", "[", "{", "«", "“", "$"}

def conll_reconstruct_text(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Deterministically reconstruct TEXT and char offsets for each token index.
    """
    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []

    pos = 0
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

        parts.append(prefix)
        pos += len(prefix)

        start = pos
        parts.append(tok)
        pos += len(tok)
        end = pos

        offsets.append((start, end))

    text = "".join(parts)
    return text, offsets

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

def predict_word_tags(tokens: List[str], tokenizer, model, id2label: Dict[int, str], device) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        pred_ids = model(**enc).logits.argmax(dim=-1).squeeze(0).tolist()

    word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=256).word_ids()
    tags = ["O"] * len(tokens)
    seen = set()
    for i, w in enumerate(word_ids):
        if w is None or w in seen:
            continue
        seen.add(w)
        tags[w] = id2label[int(pred_ids[i])]
    return tags

def spans_to_json(text: str, offsets: List[Tuple[int, int]], spans: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    ents = []
    for typ, s, e in spans:
        start = offsets[s][0]
        end = offsets[e - 1][1]
        ents.append({"type": typ, "text": text[start:end], "start": start, "end": end})
    return {"entities": ents}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=5, help="How many samples to print")
    ap.add_argument("--start", type=int, default=0, help="Start index")
    args = ap.parse_args()

    ds = load_conll()[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, from_slow=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    end = min(args.start + args.n, len(ds))
    for idx in range(args.start, end):
        ex = ds[idx]
        tokens = ex["tokens"]
        gold_tags = ex["tags"]

        text, offsets = conll_reconstruct_text(tokens)

        gold_spans = bio_to_spans(gold_tags)
        pred_tags = predict_word_tags(tokens, tokenizer, model, id2label, device)
        pred_spans = bio_to_spans(pred_tags)

        gold_json = spans_to_json(text, offsets, gold_spans)
        pred_json = spans_to_json(text, offsets, pred_spans)

        print("\n" + "=" * 110)
        print(f"SAMPLE idx={idx}  split={args.split}")
        print("TEXT:", text)
        print("GOLD_JSON:", json.dumps(gold_json, ensure_ascii=False))
        print("PRED_JSON:", json.dumps(pred_json, ensure_ascii=False))

if __name__ == "__main__":
    main()