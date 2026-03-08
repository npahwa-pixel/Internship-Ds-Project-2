# src/train_conll.py
from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

from nerjson_conll.config.labels import LABEL_LIST, LABEL2ID, ID2LABEL
from nerjson_conll.data.conll import load_conll

DEFAULT_BASE = "microsoft/deberta-v3-large"


def tokenize_and_align(ex, tokenizer, max_length: int):
    tok = tokenizer(
        ex["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = tok.word_ids()

    labels = []
    prev = None
    for w in word_ids:
        if w is None:
            labels.append(-100)
        elif w != prev:
            labels.append(LABEL2ID[ex["tags"][w]])
        else:
            labels.append(-100)
        prev = w

    tok["labels"] = labels
    return tok


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids

    true_preds, true_labels = [], []
    for pred_seq, lab_seq in zip(preds, labels):
        cur_p, cur_l = [], []
        for pr, lb in zip(pred_seq, lab_seq):
            if lb == -100:
                continue
            cur_p.append(ID2LABEL[int(pr)])
            cur_l.append(ID2LABEL[int(lb)])
        true_preds.append(cur_p)
        true_labels.append(cur_l)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds),
    }


def freeze_all_but_classifier(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    # classifier is the token-classification head
    for n, p in model.named_parameters():
        if n.startswith("classifier."):
            p.requires_grad = True


def unfreeze_last_n_layers(model, n: int):
    """
    For DeBERTa-v3 (DebertaV2ForTokenClassification), the stack is usually:
      model.deberta.encoder.layer = list of transformer layers
    """
    base = getattr(model, "deberta", None)
    if base is None:
        return

    # freeze everything first
    for _, p in model.named_parameters():
        p.requires_grad = False

    # always train classifier
    for n2, p in model.named_parameters():
        if n2.startswith("classifier."):
            p.requires_grad = True

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        layers = base.encoder.layer
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # (optional) also train the final layer norm(s) if present
    # helps a bit with stability sometimes
    for name, module in model.named_modules():
        if "LayerNorm" in module.__class__.__name__:
            for p in module.parameters():
                p.requires_grad = True


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", type=str, default=DEFAULT_BASE,
                    help="HF model name or path to a saved checkpoint directory")
    ap.add_argument("--out_dir", type=str, default="models/deberta_v3_large_conll_stage")
    ap.add_argument("--stage", type=str, choices=["head", "top4", "top8", "full"], default="top4",
                    help="head=only classifier, top4/top8=unfreeze last layers, full=all trainable")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--eval_steps", type=int, default=1200)
    ap.add_argument("--save_steps", type=int, default=1200)
    args = ap.parse_args()

    # Load dataset
    ds = load_conll()

    # Tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.init_from, use_fast=True, from_slow=True)

    train_tok = ds["train"].map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=ds["train"].column_names,
    )
    val_tok = ds["validation"].map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=ds["validation"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.init_from,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Apply stage training (memory saver)
    if args.stage == "head":
        freeze_all_but_classifier(model)
    elif args.stage == "top4":
        unfreeze_last_n_layers(model, 4)
    elif args.stage == "top8":
        unfreeze_last_n_layers(model, 8)
    elif args.stage == "full":
        for _, p in model.named_parameters():
            p.requires_grad = True

    tr, tot = count_trainable_params(model)
    print(f"[stage={args.stage}] trainable params: {tr:,} / {tot:,}")

    # IMPORTANT: Avoid MPS checkpointing bugs; OFF by default.
    # If you absolutely need it, run: ENABLE_GC=1 ...
    if os.environ.get("ENABLE_GC", "0") == "1" and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_args = TrainingArguments(
        output_dir=args.out_dir,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=50,

        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,

        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        max_grad_norm=1.0,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        report_to="none",
        dataloader_num_workers=0,

        fp16=False,
        bf16=False,

        optim="adamw_torch",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,  # ok even if deprecated warning
        data_collator=DataCollatorForTokenClassification(tokenizer),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("✅ Saved:", args.out_dir)


if __name__ == "__main__":
    main()