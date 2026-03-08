from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed
from datasets import Dataset
from nerjson.artifacts.io import write_json, write_jsonl
from nerjson.data.load import build_multidataset
from nerjson.data.spec import parse_dataset_spec
from nerjson.modeling.factory import load_model_tokencls, load_tokenizer
from nerjson.modeling.memory import pick_device, MpsEmptyCacheCallback
from nerjson.preprocessing.align import tokenize_and_align_labels
from nerjson.evaluation.metrics import compute_seqeval_metrics
from nerjson.inference.predict import predict_word_tags
from nerjson.inference.json_emit import tokens_to_json, json_validity_rate

@dataclass
class EvalConfig:
    checkpoint: str
    datasets: List[str]
    split: str = "test"
    max_length: int = 192
    batch_size: int = 4
    seed: int = 42
    trust_remote_code: bool = False
    per_dataset_eval: bool = True
    eval_json_samples: int = 200
    save_predictions_jsonl: bool = False
    out_dir: Optional[str] = None

def _tokenize(ds: Dataset, tokenizer, max_length: int):
    return ds.map(lambda ex: tokenize_and_align_labels(tokenizer, ex, max_length), batched=True, remove_columns=ds.column_names)

def run_eval(cfg: EvalConfig) -> Dict:
    set_seed(cfg.seed)
    device = pick_device()
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(cfg.checkpoint), "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)

    specs = [parse_dataset_spec(s) for s in cfg.datasets]
    train_ds, valid_ds, test_ds, bundles = build_multidataset(specs, cfg.trust_remote_code)

    eval_source = valid_ds if cfg.split == "validation" else test_ds
    if eval_source is None:
        raise ValueError(f"No split={cfg.split} available across selected datasets.")

    tokenizer = load_tokenizer(cfg.checkpoint)
    model = load_model_tokencls(cfg.checkpoint, attn_eager=True)

    eval_tok = _tokenize(eval_source, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "_eval_tmp"),
        per_device_eval_batch_size=cfg.batch_size,
        report_to=[],
        fp16=False, bf16=False,
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
    metrics_per_dataset = None
    if cfg.per_dataset_eval:
        metrics_per_dataset = {}
        for b in bundles:
            split_ds = b.test if cfg.split == "test" else b.valid
            if split_ds is None:
                continue
            tok = _tokenize(split_ds, tokenizer, cfg.max_length)
            metrics_per_dataset[b.spec.display] = trainer.evaluate(tok)

    n = min(cfg.eval_json_samples, len(eval_source))
    idxs = list(range(len(eval_source))); random.shuffle(idxs); idxs = idxs[:n]
    model.to(device); model.eval()
    samples = []
    for i in idxs:
        toks = eval_source[i]["tokens"]
        pred_tags = predict_word_tags(toks, tokenizer, model, device, cfg.max_length)
        samples.append((toks, pred_tags))
    jv = json_validity_rate(samples)

    pred_jsonl_path = None
    if cfg.save_predictions_jsonl:
        pred_jsonl_path = os.path.join(out_dir, "predictions_sample.jsonl")
        write_jsonl(pred_jsonl_path, [tokens_to_json(t, p) for t, p in samples])

    summary = {
        "checkpoint": cfg.checkpoint,
        "datasets": cfg.datasets,
        "split": cfg.split,
        "metrics_concat": metrics_concat,
        "metrics_per_dataset": metrics_per_dataset,
        "json_validity_sampled": jv,
        "json_validity_samples": n,
        "device": device,
        "predictions_sample_jsonl": pred_jsonl_path,
        "out_dir": out_dir,
    }
    write_json(os.path.join(out_dir, "eval_summary.json"), summary)
    write_json(os.path.join(out_dir, "metrics_concat.json"), metrics_concat)
    if metrics_per_dataset is not None:
        write_json(os.path.join(out_dir, "metrics_per_dataset.json"), metrics_per_dataset)
    return summary
