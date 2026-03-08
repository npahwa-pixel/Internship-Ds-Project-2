from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed

from nerjson.artifacts.io import write_json, write_jsonl
from nerjson.data.load import build_multidataset, DatasetBundle
from nerjson.inference.json_emit import json_validity_rate, tokens_to_json
from nerjson.inference.predict import predict_word_tags
from nerjson.modeling.factory import load_model_tokencls, load_tokenizer
from nerjson.modeling.memory import pick_device, MpsEmptyCacheCallback
from nerjson.preprocessing.align import tokenize_and_align_labels
from nerjson.evaluation.metrics import compute_seqeval_metrics


@dataclass
class EvalConfig:
    checkpoint: str
    datasets: List[str]
    split: str = "test"  # test|validation
    max_length: int = 192
    batch_size: int = 4
    seed: int = 42
    trust_remote_code: bool = False

    per_dataset_eval: bool = True
    eval_json_samples: int = 200
    save_predictions_jsonl: bool = False

    out_dir: Optional[str] = None  # if None, writes next to checkpoint


def _resolve_out_dir(cfg: EvalConfig) -> str:
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        return cfg.out_dir
    # default: write into a safe folder next to checkpoint (avoid polluting checkpoint dir)
    p = cfg.checkpoint
    base = os.path.dirname(p) if os.path.basename(p).startswith("checkpoint-") else p
    out = os.path.join(base, "eval_outputs")
    os.makedirs(out, exist_ok=True)
    return out


def _tokenize_split(ds: Dataset, tokenizer, max_length: int):
    return ds.map(
        lambda ex: tokenize_and_align_labels(tokenizer, ex, max_length),
        batched=True,
        remove_columns=ds.column_names,
    )


def run_eval(cfg: EvalConfig) -> Dict:
    set_seed(cfg.seed)
    device = pick_device()
    out_dir = _resolve_out_dir(cfg)

    train_ds, valid_ds, test_ds, bundles = build_multidataset(cfg.datasets, trust_remote_code=cfg.trust_remote_code)

    # Split selection
    if cfg.split == "validation":
        eval_source = valid_ds
    else:
        if test_ds is None:
            raise ValueError("Requested split=test but no test split available in selected datasets.")
        eval_source = test_ds

    tokenizer = load_tokenizer(cfg.checkpoint)
    model = load_model_tokencls(cfg.checkpoint, attn_eager=True)

    eval_tok = _tokenize_split(eval_source, tokenizer, cfg.max_length)

    # Minimal TrainingArguments for evaluation
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "_eval_tmp"),
        per_device_eval_batch_size=cfg.batch_size,
        report_to=[],
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_seqeval_metrics,
        callbacks=[MpsEmptyCacheCallback(every_n_steps=50)],
    )

    metrics_concat = trainer.evaluate()
    print("Concat metrics:", metrics_concat)

    metrics_per_dataset = None
    if cfg.per_dataset_eval:
        metrics_per_dataset = {}
        for b in bundles:
            split_ds = b.test if cfg.split == "test" else b.valid
            if split_ds is None:
                continue
            tok = _tokenize_split(split_ds, tokenizer, cfg.max_length)
            m = trainer.evaluate(tok)
            metrics_per_dataset[b.name] = m
        print("Per-dataset metrics computed for:", list(metrics_per_dataset.keys()))

    # JSON validity sampling (uses deterministic token offsets via dataset tokens)
    n = min(cfg.eval_json_samples, len(eval_source))
    idxs = list(range(len(eval_source)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    model.to(device)
    model.eval()

    samples = []
    for i in idxs:
        toks = eval_source[i]["tokens"]
        pred_tags = predict_word_tags(toks, tokenizer, model, device, cfg.max_length)
        samples.append((toks, pred_tags))

    jv = json_validity_rate(samples)
    print({"json_validity": jv, "samples": n})

    pred_jsonl_path = None
    if cfg.save_predictions_jsonl:
        pred_jsonl_path = os.path.join(out_dir, "predictions_sample.jsonl")
        write_jsonl(pred_jsonl_path, [tokens_to_json(t, p) for t, p in samples])
        print("Saved sample JSONL predictions to:", pred_jsonl_path)

    summary = {
        "checkpoint": cfg.checkpoint,
        "datasets": cfg.datasets,
        "split": cfg.split,
        "metrics_concat": metrics_concat,
        "metrics_per_dataset": metrics_per_dataset,
        "json_validity_sampled": jv,
        "json_validity_samples": n,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "device": device,
        "predictions_sample_jsonl": pred_jsonl_path,
    }

    write_json(os.path.join(out_dir, "eval_summary.json"), summary)
    write_json(os.path.join(out_dir, "metrics_concat.json"), metrics_concat)
    if metrics_per_dataset is not None:
        write_json(os.path.join(out_dir, "metrics_per_dataset.json"), metrics_per_dataset)

    return summary
