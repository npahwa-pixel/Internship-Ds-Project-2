from __future__ import annotations
import argparse, os, yaml
from nerjson.evaluation.runner import EvalConfig, run_eval
from nerjson.modeling.resolve import resolve_checkpoint_path

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    p = argparse.ArgumentParser(description="Evaluate on 4 datasets (concat + per-dataset + JSON validity).")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default="models/deberta_fullft_multi_4ds")
    p.add_argument("--datasets", nargs="+", default=["tner/conll2003","tner/ontonotes5","tner/wikiann:en","tner/btc"])
    p.add_argument("--split", choices=["validation","test"], default="test")
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--eval_json_samples", type=int, default=200)
    p.add_argument("--save_predictions_jsonl", action="store_true")
    p.add_argument("--per_dataset_eval", action="store_true")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    cfg_dict = _load_yaml(args.config) if args.config else {}
    raw_ckpt = cfg_dict.get("checkpoint", args.checkpoint)
    ckpt = resolve_checkpoint_path(raw_ckpt) if os.path.exists(os.path.expanduser(raw_ckpt)) else raw_ckpt

    cfg = EvalConfig(
        checkpoint=ckpt,
        datasets=cfg_dict.get("datasets", args.datasets),
        split=cfg_dict.get("split", args.split),
        max_length=int(cfg_dict.get("max_length", args.max_length)),
        batch_size=int(cfg_dict.get("batch_size", args.batch_size)),
        seed=int(cfg_dict.get("seed", args.seed)),
        trust_remote_code=bool(cfg_dict.get("trust_remote_code", args.trust_remote_code)),
        eval_json_samples=int(cfg_dict.get("eval_json_samples", args.eval_json_samples)),
        save_predictions_jsonl=bool(cfg_dict.get("save_predictions_jsonl", args.save_predictions_jsonl)),
        per_dataset_eval=bool(cfg_dict.get("per_dataset_eval", args.per_dataset_eval)),
        out_dir=cfg_dict.get("out_dir", args.out_dir),
    )
    run_eval(cfg)

if __name__ == "__main__":
    main()
