from __future__ import annotations

import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from nerjson.artifacts.io import write_json, write_jsonl
from nerjson.data.load import build_multidataset
from nerjson.evaluation.metrics import compute_seqeval_metrics
from nerjson.inference.json_emit import json_validity_rate, tokens_to_json
from nerjson.inference.predict import predict_word_tags
from nerjson.modeling.factory import load_model_tokencls, load_tokenizer
from nerjson.modeling.memory import count_trainable_params, freeze_bottom_layers, pick_device, MpsEmptyCacheCallback
from nerjson.preprocessing.align import tokenize_and_align_labels
from nerjson.training.args import TrainConfig


def _tokenize(ds, tokenizer, max_length: int):
    return ds.map(
        lambda ex: tokenize_and_align_labels(tokenizer, ex, max_length),
        batched=True,
        remove_columns=ds.column_names,
    )


def run_train(cfg: TrainConfig) -> Dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = pick_device()
    print("Device:", device)
    print("max_length:", cfg.max_length, "| batch:", cfg.batch_size, "| grad_accum:", cfg.grad_accum)
    print("Loading datasets:", cfg.datasets)

    train_ds, valid_ds, test_ds, bundles = build_multidataset(cfg.datasets, trust_remote_code=cfg.trust_remote_code)

    tokenizer = load_tokenizer(cfg.model)

    train_tok = _tokenize(train_ds, tokenizer, cfg.max_length)
    valid_tok = _tokenize(valid_ds, tokenizer, cfg.max_length)

    model = load_model_tokencls(cfg.model, attn_eager=True)

    freeze_bottom_layers(model, cfg.freeze_bottom_layers)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,

        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="linear",

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_steps=50,
        logging_first_step=True,
        report_to=[],

        fp16=False,
        bf16=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        optim="adafactor",
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_seqeval_metrics,
        callbacks=[MpsEmptyCacheCallback(every_n_steps=cfg.mps_empty_cache_steps)],
    )

    print("\n=== Training (fine-tune) ===")
    trainer.train()

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

    print("\n=== Eval on CONCAT validation ===")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    test_metrics = None
    if test_ds is not None:
        test_tok = _tokenize(test_ds, tokenizer, cfg.max_length)
        print("\n=== Eval on CONCAT test ===")
        test_metrics = trainer.evaluate(test_tok)
        print(test_metrics)

    # Optional per-dataset evaluation
    per_dataset_metrics = None
    if cfg.per_dataset_eval:
        per_dataset_metrics = {}
        for b in bundles:
            if b.valid is not None:
                tok = _tokenize(b.valid, tokenizer, cfg.max_length)
                per_dataset_metrics.setdefault(b.name, {})["validation"] = trainer.evaluate(tok)
            if b.test is not None:
                tok = _tokenize(b.test, tokenizer, cfg.max_length)
                per_dataset_metrics.setdefault(b.name, {})["test"] = trainer.evaluate(tok)

    # JSON validity sampling
    eval_source = test_ds if test_ds is not None else valid_ds
    n = min(cfg.eval_json_samples, len(eval_source))
    idxs = list(range(len(eval_source)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    trainer.model.to(device)
    trainer.model.eval()

    samples = []
    for i in idxs:
        toks = eval_source[i]["tokens"]
        pred_tags = predict_word_tags(toks, tokenizer, trainer.model, device, cfg.max_length)
        samples.append((toks, pred_tags))

    jv = json_validity_rate(samples)
    print(f"\n=== JSON validity on {n} samples ===")
    print({"json_validity": jv})

    pred_jsonl_path = None
    if cfg.save_predictions_jsonl:
        pred_jsonl_path = os.path.join(cfg.output_dir, "predictions_sample.jsonl")
        write_jsonl(pred_jsonl_path, [tokens_to_json(t, p) for t, p in samples])
        print("Saved sample JSONL predictions to:", pred_jsonl_path)

    summary = {
        "model": cfg.model,
        "datasets": cfg.datasets,
        "val_metrics_concat": val_metrics,
        "test_metrics_concat": test_metrics,
        "metrics_per_dataset": per_dataset_metrics,
        "json_validity_sampled": jv,
        "json_validity_samples": n,
        "output_dir": cfg.output_dir,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "grad_accum": cfg.grad_accum,
        "freeze_bottom_layers": cfg.freeze_bottom_layers,
        "optim": "adafactor",
        "predictions_sample_jsonl": pred_jsonl_path,
        "device": device,
        "best_model_checkpoint": best_ckpt,
    }

    write_json(os.path.join(cfg.output_dir, "run_summary.json"), summary)
    write_json(os.path.join(cfg.output_dir, "resolved_train_config.json"), asdict(cfg))

    print("\n✅ Done. Best checkpoint:", best_ckpt or "(unknown)")
    return summary
