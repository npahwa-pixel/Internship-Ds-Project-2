from __future__ import annotations

import argparse
from typing import Dict, Any, Callable, Tuple, Optional

import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

from peft import PeftConfig, PeftModel

from build_external_datasets import load_named_dataset


def tokenize_and_align(
    ex: Dict[str, Any],
    tokenizer,
    max_length: int,
    label2id: Dict[str, int],
) -> Dict[str, Any]:
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
            labels.append(label2id[ex["tags"][w]])
        else:
            labels.append(-100)
        prev = w

    tok["labels"] = labels
    return tok


def make_compute_metrics(id2label: Dict[int, str]) -> Callable:
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        true_preds, true_labels = [], []
        for pred_seq, lab_seq in zip(preds, labels):
            cur_p, cur_l = [], []
            for pr, lb in zip(pred_seq, lab_seq):
                if lb == -100:
                    continue
                cur_p.append(id2label[int(pr)])
                cur_l.append(id2label[int(lb)])
            true_preds.append(cur_p)
            true_labels.append(cur_l)

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
            "accuracy": accuracy_score(true_labels, true_preds),
        }

    return compute_metrics


def _normalize_config_maps(model) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = dict(model.config.label2id)

    raw_id2label = dict(model.config.id2label)
    id2label: Dict[int, str] = {}
    for k, v in raw_id2label.items():
        id2label[int(k)] = str(v) if not isinstance(k, int) else str(v)
    # If keys already int, above int(k) works too.
    return label2id, id2label


def load_model_and_tokenizer(base_model: Optional[str], lora_dir: Optional[str]):
    """
    - If lora_dir is provided: load base from adapter config and attach adapter.
    - Else: load base_model directly.
    """
    if lora_dir:
        peft_cfg = PeftConfig.from_pretrained(lora_dir)
        base_id = peft_cfg.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, from_slow=True)
        base = AutoModelForTokenClassification.from_pretrained(base_id)
        model = PeftModel.from_pretrained(base, lora_dir)
        return tokenizer, model

    if not base_model:
        raise ValueError("base_model is required when --lora_dir is not provided.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, from_slow=True)
    model = AutoModelForTokenClassification.from_pretrained(base_model)
    return tokenizer, model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True, choices=["conll", "wikiann", "ontonotes5", "wnut17"])
    ap.add_argument("--lang", type=str, default="en", help="Only used for WikiANN")
    ap.add_argument("--split", type=str, default="test", choices=["validation", "test"])

    ap.add_argument("--base_model", type=str, default=None, help="HF id or local dir of TokenClassification model")
    ap.add_argument("--lora_dir", type=str, default=None, help="PEFT adapter dir (optional)")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--print_report", action="store_true")

    args = ap.parse_args()

    ds = load_named_dataset(args.dataset, lang=args.lang)[args.split]

    tokenizer, model = load_model_and_tokenizer(args.base_model, args.lora_dir)
    label2id, id2label = _normalize_config_maps(model)

    # sanity: ensure dataset tags exist in model label space
    sample_tags = set()
    for i in range(min(200, len(ds))):
        sample_tags.update(ds[i]["tags"])

    missing = sorted([t for t in sample_tags if t not in label2id])
    if missing:
        raise ValueError(
            "Dataset tag-space != model label2id.\n"
            f"Missing tags (first 25): {missing[:25]}\n"
            "This usually happens if your model wasn't trained with CoNLL BIO labels."
        )

    tok_ds = ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, args.max_length, label2id),
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
        compute_metrics=make_compute_metrics(id2label),
    )

    metrics = trainer.evaluate(tok_ds)

    title = f"{args.dataset.upper()}" + (f"[{args.lang}]" if args.dataset == "wikiann" else "")
    mode = "LoRA" if args.lora_dir else "BASE"

    print(f"\n=== {title} {args.split.upper()} METRICS ({mode}) ===")
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
                cur_p.append(id2label[int(pr)])
                cur_l.append(id2label[int(lb)])
            true_preds.append(cur_p)
            true_labels.append(cur_l)

        print("\n=== SEQEVAL REPORT ===")
        print(classification_report(true_labels, true_preds, digits=4))


if __name__ == "__main__":
    main()