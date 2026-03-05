# src/train_conll_lora.py
from __future__ import annotations

import argparse
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

from peft import LoraConfig, get_peft_model, TaskType

from build_conll import load_conll
from labels import LABEL_LIST, LABEL2ID, ID2LABEL


def parse_target_modules(s: str) -> list[str]:
    mods = [m.strip() for m in s.split(",") if m.strip()]
    if not mods:
        raise ValueError("--target_modules cannot be empty")
    return mods


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base_model",
        type=str,
        default="tner/deberta-v3-large-conll2003",
        help="Base token-classification model to adapt with LoRA",
    )
    ap.add_argument("--out_dir", type=str, default="models/deberta_conll_lora")

    # data/tokenization
    ap.add_argument("--max_length", type=int, default=256)

    # training
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_bs", type=int, default=1)
    ap.add_argument("--eval_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=800)
    ap.add_argument("--save_steps", type=int, default=800)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine"])
    ap.add_argument("--patience", type=int, default=2)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="query_proj,key_proj,value_proj",
        help="Comma-separated module names for LoRA (e.g. query_proj,key_proj,value_proj,dense)",
    )
    ap.add_argument(
        "--train_classifier",
        action="store_true",
        help="Also train the classification head weights",
    )

    args = ap.parse_args()

    # Load dataset
    ds = load_conll()

    # IMPORTANT: fast tokenizer required for word_ids()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, from_slow=True)

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

    # Load base model
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # LoRA setup
    target_modules = parse_target_modules(args.target_modules)
    print("LoRA target_modules:", target_modules)

    lora_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # Optionally train classifier head too (usually good)
    if args.train_classifier:
        for n, p in model.named_parameters():
            if "classifier" in n:
                p.requires_grad = True

    # Print trainable parameters (sanity)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # Training arguments (stable)
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
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        max_grad_norm=0.5,
        adam_epsilon=1e-6,

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
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("✅ Saved LoRA model to:", args.out_dir)


if __name__ == "__main__":
    main()