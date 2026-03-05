# src/eval_conll.py
from __future__ import annotations

import argparse
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

from build_conll import load_conll
from labels import LABEL2ID, ID2LABEL


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
    ap.add_argument("--model_dir", type=str, required=True, help="Local folder or checkpoint folder")
    ap.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--print_report", action="store_true")
    args = ap.parse_args()

    ds = load_conll()[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    tok_ds = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=ds.column_names,
    )

    targs = TrainingArguments(
        output_dir="models/_tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(tok_ds)

    print(f"\n=== CoNLL2003 {args.split.upper()} METRICS ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v}")

    if args.print_report:
        pred = trainer.predict(tok_ds)
        preds = np.argmax(pred.predictions, axis=-1)
        labels = pred.label_ids

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

        print("\n=== SEQEVAL REPORT ===")
        print(classification_report(true_labels, true_preds, digits=4))


if __name__ == "__main__":
    main()