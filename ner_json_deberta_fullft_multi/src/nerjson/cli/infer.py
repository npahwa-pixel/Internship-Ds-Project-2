from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import yaml

from nerjson.artifacts.io import write_jsonl, write_json
from nerjson.inference.predict import predict_word_tags
from nerjson.inference.json_emit import tokens_to_json
from nerjson.modeling.factory import load_model_tokencls, load_tokenizer
from nerjson.modeling.memory import pick_device
from nerjson.modeling.resolve import resolve_checkpoint_path


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_texts(text: Optional[str], input_file: Optional[str]) -> Iterable[str]:
    if text is not None:
        yield text
        return
    if input_file is not None:
        p = Path(input_file)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield line
        return
    raise ValueError("Provide either --text or --input_file")


def main() -> None:
    p = argparse.ArgumentParser(description="Inference: text → NER JSON (no UI).")
    p.add_argument("--config", default=None, help="Optional YAML config path.")
    p.add_argument("--checkpoint", default="models/deberta_fullft_multi",
                   help="Local checkpoint dir OR run dir containing checkpoint-* subdirs.")
    p.add_argument("--max_length", type=int, default=192)

    p.add_argument("--text", default=None)
    p.add_argument("--input_file", default=None)
    p.add_argument("--output_jsonl", default="predictions.jsonl")
    p.add_argument("--save_meta", action="store_true")

    args = p.parse_args()
    print("CWD:", os.getcwd())

    cfg = {}
    if args.config:
        cfg.update(_load_yaml(args.config))

    raw_ckpt = cfg.get("checkpoint", args.checkpoint)

    if os.path.exists(os.path.expanduser(raw_ckpt)):
        checkpoint = resolve_checkpoint_path(raw_ckpt)
    else:
        checkpoint = raw_ckpt  # HF model id

    max_length = int(cfg.get("max_length", args.max_length))
    text = cfg.get("input_text", args.text)
    input_file = cfg.get("input_file", args.input_file)
    output_jsonl = cfg.get("output_jsonl", args.output_jsonl)

    print("Checkpoint (resolved):", checkpoint)

    device = pick_device()
    tokenizer = load_tokenizer(checkpoint)
    model = load_model_tokencls(checkpoint, attn_eager=True)
    model.to(device)
    model.eval()

    outputs = []
    for t in _iter_texts(text, input_file):
        tokens = t.split()
        pred_tags = predict_word_tags(tokens, tokenizer, model, device, max_length)
        obj = tokens_to_json(tokens, pred_tags)
        if args.save_meta:
            obj["_meta"] = {"text": t, "tokens": tokens, "tags": pred_tags}
        outputs.append(obj)

    write_jsonl(output_jsonl, outputs)
    print("Wrote:", output_jsonl)

    if args.save_meta:
        write_json("infer_meta.json", {"checkpoint": checkpoint, "device": device, "max_length": max_length})


if __name__ == "__main__":
    main()
