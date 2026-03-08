# DeBERTa Full Fine-tune — Proper ML Pipeline (No UI)

This repo refactors a single-file training script into a **clean end-to-end ML workflow**:

**data ingestion → tag unification → preprocessing → training → evaluation → inference → artifacts**

It is designed to work with an existing checkpoint directory like:
`deberta_fullft_multi/checkpoint-14319/`

## Project layout

```text
.
├── train.py
├── evaluate.py
├── infer.py
├── requirements.txt
├── configs/
└── src/nerjson/
    ├── config/          # labels + JSON schema
    ├── data/            # HF loading + T-NER label mapping + unify tags
    ├── preprocessing/   # tokenize+align (single source of truth)
    ├── modeling/        # model factory + memory knobs (MPS helpers, freezing)
    ├── training/        # training runner
    ├── evaluation/      # metrics + evaluation runner (concat + per-dataset)
    ├── inference/       # predict tags + BIO→spans + JSON emission
    └── artifacts/       # write run_summary, metrics, predictions jsonl
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Tip (recommended): run via `PYTHONPATH=src` so the package imports cleanly.

## Train (multi-dataset)

```bash
PYTHONPATH=src python train.py   --model microsoft/deberta-v3-large   --datasets tner/conll2003 tner/ontonotes5 tner/wnut2017   --output_dir deberta_fullft_multi   --max_length 192   --epochs 3 --lr 2e-5 --weight_decay 0.01   --batch_size 1 --eval_batch_size 1 --grad_accum 16   --warmup_ratio 0.1   --freeze_bottom_layers 12   --save_predictions_jsonl   --per_dataset_eval
```

Artifacts written under `output_dir/`:
- `run_summary.json`
- `metrics_concat.json` (validation/test if available)
- `metrics_per_dataset.json` (if enabled)
- `predictions_sample.jsonl` (if enabled)
- HF checkpoints (`checkpoint-*`)

## Evaluate (from an existing checkpoint)

```bash
PYTHONPATH=src python evaluate.py   --checkpoint deberta_fullft_multi/checkpoint-14319   --datasets tner/conll2003 tner/ontonotes5 tner/wnut2017   --split test   --batch_size 4   --max_length 192   --save_predictions_jsonl   --per_dataset_eval
```

This writes `eval_summary.json`, `metrics_concat.json`, and optionally `metrics_per_dataset.json` next to the checkpoint (or into `--out_dir` if provided).

## Inference (NER → JSON)

```bash
PYTHONPATH=src python infer.py   --checkpoint deberta_fullft_multi/checkpoint-14319   --text "OpenAI is based in San Francisco."   --max_length 192
```

Or from a file (one text per line):

```bash
PYTHONPATH=src python infer.py   --checkpoint deberta_fullft_multi/checkpoint-14319   --input_file texts.txt   --output_jsonl predictions.jsonl
```

### Note on offsets
For deterministic behavior matching training/eval, offsets are computed on `" ".join(tokens)` where `tokens = text.split()`.
This makes offsets reproducible and consistent with dataset-style tokenization.

## Config files (optional)
You can also use YAML configs under `configs/` and pass `--config configs/train.yaml` to each entrypoint.



## Common issue: checkpoint path treated as Hugging Face repo id

If you see an error like `... is not a local folder and is not a valid model identifier`, it means the checkpoint path you passed **does not exist from your current working directory**. Run `pwd` and `ls` to verify, then pass an absolute path or correct relative path.

Examples:

```bash
PYTHONPATH=src python evaluate.py --checkpoint ../deberta_fullft_multi/checkpoint-14319
PYTHONPATH=src python infer.py --checkpoint /absolute/path/to/deberta_fullft_multi/checkpoint-14319 --text "OpenAI ..."
```

You can also pass the run directory and the code will auto-pick the latest/best `checkpoint-*`:

```bash
PYTHONPATH=src python evaluate.py --checkpoint ../deberta_fullft_multi
```


## Models folder (your big artifacts live here)

Put your existing directory here (copy as a folder):

```text
models/deberta_fullft_multi/
  checkpoint-14319/
  run_summary.json
  predictions_sample.jsonl
```

Then run:

```bash
PYTHONPATH=src python evaluate.py --checkpoint models/deberta_fullft_multi --split test --per_dataset_eval --save_predictions_jsonl
PYTHONPATH=src python infer.py --checkpoint models/deberta_fullft_multi --text "OpenAI is based in San Francisco." --output_jsonl outputs/infer_outputs/predictions.jsonl
```
