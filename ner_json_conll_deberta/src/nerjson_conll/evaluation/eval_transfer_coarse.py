# src/eval_transfer_coarse.py
from __future__ import annotations

import argparse
import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# Your repo dataset loader (uses build_ontonotes5/wikiann/wnut17 etc.)
from nerjson_conll.data.registry import load_named_dataset


# Prefer your repo's label space if available (CoNLL-style 9 tags)
try:
    from nerjson_conll.config.labels import ID2LABEL, LABEL2ID  # type: ignore
except Exception:
    _LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    LABEL2ID = {t: i for i, t in enumerate(_LABELS)}
    ID2LABEL = {i: t for t, i in LABEL2ID.items()}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(
    model_dir: Optional[str],
    base_model: Optional[str],
    lora_dir: Optional[str],
):
    """
    - If lora_dir is provided: load base model as TokenClassification with correct num_labels, then apply adapter via PEFT.
    - Else: load token-classification model from model_dir.
    """
    if lora_dir:
        from peft import PeftModel, PeftConfig  # requires peft installed

        peft_cfg = PeftConfig.from_pretrained(lora_dir)
        base_id = base_model or peft_cfg.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

        # CRITICAL: base model head must match adapter's label space (your 9-tag CoNLL BIO space)
        base = AutoModelForTokenClassification.from_pretrained(
            base_id,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model = PeftModel.from_pretrained(base, lora_dir)
        return model, tokenizer

    if not model_dir:
        raise ValueError("Provide --model_dir for head model OR --lora_dir for adapter model.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model, tokenizer


def make_tokenize_and_align(tokenizer, model_label2id: Dict[str, int], max_length: int):
    """
    Align word-level dataset tags to token-level labels for Trainer.
    Dataset tags can be:
      - ints (ids in LABEL2ID space), OR
      - strings (label names).
    We map by label name into the model's label2id.
    """
    def fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        labels_out: List[List[int]] = []
        for i in range(len(examples["tokens"])):
            word_ids = enc.word_ids(batch_index=i)
            word_tags = examples["tags"][i]

            aligned: List[int] = []
            prev = None
            for wi in word_ids:
                if wi is None:
                    aligned.append(-100)
                elif wi != prev:
                    tag = word_tags[wi]
                    if isinstance(tag, (int, np.integer)):
                        name = ID2LABEL[int(tag)]
                    else:
                        name = str(tag)

                    aligned.append(int(model_label2id.get(name, model_label2id.get("O", 0))))
                else:
                    aligned.append(-100)
                prev = wi

            labels_out.append(aligned)

        enc["labels"] = labels_out
        return enc

    return fn


def compute_seqeval(model_id2label: Dict[int, str]):
    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        y_true, y_pred = [], []
        for p, l in zip(preds, labels):
            t_seq, p_seq = [], []
            for pi, li in zip(p, l):
                if li == -100:
                    continue
                t_seq.append(model_id2label[int(li)])
                p_seq.append(model_id2label[int(pi)])
            y_true.append(t_seq)
            y_pred.append(p_seq)

        return {
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }

    return fn


def main():
    ap = argparse.ArgumentParser(
        "Cross-dataset transfer eval in CoNLL label space (PER/ORG/LOC/MISC BIO)"
    )
    ap.add_argument("--dataset", required=True, choices=["conll", "ontonotes5", "wikiann", "wnut17"])
    ap.add_argument("--lang", default="en", help="Only used for WikiANN")
    ap.add_argument("--split", default="test", choices=["validation", "test"])

    # Head model
    ap.add_argument("--model_dir", default=None, help="TokenClassification checkpoint dir (head model)")

    # LoRA adapter
    ap.add_argument("--base_model", default=None, help="HF base id or local dir (for LoRA)")
    ap.add_argument("--lora_dir", default=None, help="PEFT adapter dir (optional)")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--print_report", action="store_true")
    args = ap.parse_args()

    if not args.model_dir and not args.lora_dir:
        raise ValueError("Provide either --model_dir (head) or --lora_dir (adapter).")

    # Load dataset split (builders should output tokens + tags in LABEL2ID space)
    dsd = load_named_dataset(args.dataset, lang=args.lang)
    if args.split not in dsd:
        raise ValueError(f"Split {args.split} not found for dataset={args.dataset}. Available={list(dsd.keys())}")
    ds = dsd[args.split]

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model, args.lora_dir)

    device = pick_device()
    model.to(device)
    model.eval()

    # Model label mapping
    model_label2id = getattr(model.config, "label2id", None) or LABEL2ID
    model_id2label = getattr(model.config, "id2label", None)
    if not model_id2label:
        model_id2label = {v: k for k, v in model_label2id.items()}
    model_id2label = {int(k): str(v) for k, v in model_id2label.items()}

    # Tokenize & align
    tok_fn = make_tokenize_and_align(tokenizer, model_label2id, args.max_length)
    tok_ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # Eval
    targs = TrainingArguments(
        output_dir="models/_tmp_eval_transfer",
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=tok_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_seqeval(model_id2label),
    )

    metrics = trainer.evaluate()
    print(f"\n=== {args.dataset.upper()} {args.split.upper()} METRICS (TRANSFER-COARSE) ===")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")

    if args.print_report:
        pred = trainer.predict(tok_ds)
        preds = pred.predictions
        labels = pred.label_ids
        pred_ids = np.argmax(preds, axis=-1)

        y_true, y_pred = [], []
        for p, l in zip(pred_ids, labels):
            t_seq, p_seq = [], []
            for pi, li in zip(p, l):
                if li == -100:
                    continue
                t_seq.append(model_id2label[int(li)])
                p_seq.append(model_id2label[int(pi)])
            y_true.append(t_seq)
            y_pred.append(p_seq)

        print("\n=== SEQEVAL REPORT (TRANSFER-COARSE) ===")
        print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
