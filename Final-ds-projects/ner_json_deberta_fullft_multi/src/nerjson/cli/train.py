from __future__ import annotations

import argparse
import yaml

from nerjson.training.args import TrainConfig
from nerjson.training.runner import run_train


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeBERTa token-classifier on unified NER datasets (no UI).")
    p.add_argument("--config", help="Optional YAML config path.", default=None)

    p.add_argument("--model", default="microsoft/deberta-v3-large")
    p.add_argument("--datasets", nargs="+", default=["tner/conll2003", "tner/ontonotes5", "tner/wnut2017"])
    p.add_argument("--output_dir", default="models/deberta_fullft_multi")

    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)

    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--freeze_bottom_layers", type=int, default=12)
    p.add_argument("--mps_empty_cache_steps", type=int, default=20)

    p.add_argument("--eval_json_samples", type=int, default=200)
    p.add_argument("--save_predictions_jsonl", action="store_true")
    p.add_argument("--per_dataset_eval", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    args = p.parse_args()

    cfg_dict = {}
    if args.config:
        cfg_dict.update(_load_yaml(args.config))

    # CLI overrides YAML
    cfg = TrainConfig(
        model=cfg_dict.get("model", args.model),
        datasets=cfg_dict.get("datasets", args.datasets),
        output_dir=cfg_dict.get("output_dir", args.output_dir),
        max_length=int(cfg_dict.get("max_length", args.max_length)),
        epochs=int(cfg_dict.get("epochs", args.epochs)),
        lr=float(cfg_dict.get("lr", args.lr)),
        weight_decay=float(cfg_dict.get("weight_decay", args.weight_decay)),
        batch_size=int(cfg_dict.get("batch_size", args.batch_size)),
        eval_batch_size=int(cfg_dict.get("eval_batch_size", args.eval_batch_size)),
        grad_accum=int(cfg_dict.get("grad_accum", args.grad_accum)),
        warmup_ratio=float(cfg_dict.get("warmup_ratio", args.warmup_ratio)),
        seed=int(cfg_dict.get("seed", args.seed)),
        freeze_bottom_layers=int(cfg_dict.get("freeze_bottom_layers", args.freeze_bottom_layers)),
        mps_empty_cache_steps=int(cfg_dict.get("mps_empty_cache_steps", args.mps_empty_cache_steps)),
        eval_json_samples=int(cfg_dict.get("eval_json_samples", args.eval_json_samples)),
        save_predictions_jsonl=bool(cfg_dict.get("save_predictions_jsonl", args.save_predictions_jsonl)),
        per_dataset_eval=bool(cfg_dict.get("per_dataset_eval", args.per_dataset_eval)),
        trust_remote_code=bool(cfg_dict.get("trust_remote_code", args.trust_remote_code)),
    )

    run_train(cfg)


if __name__ == "__main__":
    main()
