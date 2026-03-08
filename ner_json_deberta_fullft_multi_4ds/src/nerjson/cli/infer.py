from __future__ import annotations
import argparse, os, yaml
from pathlib import Path
from typing import Iterable, Optional
from nerjson.artifacts.io import write_jsonl
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
        yield text; return
    if input_file is not None:
        for line in Path(input_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield line
        return
    raise ValueError("Provide --text or --input_file")

def main():
    p = argparse.ArgumentParser(description="Inference: text → JSONL (4-dataset model).")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/deberta_fullft_multi_4ds")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--text", default=None)
    p.add_argument("--input_file", default=None)
    p.add_argument("--output_jsonl", default="outputs/infer_outputs/predictions.jsonl")
    args = p.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}
    raw_ckpt = cfg.get("checkpoint", args.checkpoint)
    ckpt = resolve_checkpoint_path(raw_ckpt) if os.path.exists(os.path.expanduser(raw_ckpt)) else raw_ckpt

    max_length = int(cfg.get("max_length", args.max_length))
    text = cfg.get("input_text", args.text)
    input_file = cfg.get("input_file", args.input_file)
    out = cfg.get("output_jsonl", args.output_jsonl)

    device = pick_device()
    tokenizer = load_tokenizer(ckpt)
    model = load_model_tokencls(ckpt, attn_eager=True)
    model.to(device); model.eval()

    outputs = []
    for t in _iter_texts(text, input_file):
        toks = t.split()
        tags = predict_word_tags(toks, tokenizer, model, device, max_length)
        outputs.append(tokens_to_json(toks, tags))
    write_jsonl(out, outputs)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
