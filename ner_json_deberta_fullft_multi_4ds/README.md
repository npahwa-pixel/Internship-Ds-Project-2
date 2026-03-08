# DeBERTa NER→JSON — 4-dataset ML Pipeline

This project is a **proper ML pipeline structure** for your **4-dataset** run:
- `tner/conll2003`
- `tner/ontonotes5`
- `tner/wikiann:en`
- `tner/btc`

It supports:
- unified BIO tags (PER/ORG/LOC/MISC)
- canonicalized dataset schema for safe concatenation across datasets
- training + concat eval + per-dataset eval
- deterministic NER→JSON emission + JSON validity rate sampling
- **NO UI** (models are loaded from a local folder)

## Where to copy your model folder (big artifacts)

```
https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0
```

## Install

If pip isn’t present in the venv:
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

Then:
```bash
python -m pip install -r requirements.txt
```

## Evaluate your existing checkpoint/run dir

Recommended: pass the RUN DIR and it will auto-pick best/latest `checkpoint-*`:

```bash
PYTHONPATH=src python evaluate.py --config configs/eval.yaml
```

Or explicit CLI:

```bash
PYTHONPATH=src python evaluate.py \
  --checkpoint models/deberta_fullft_multi_4ds \
  --datasets tner/conll2003 tner/ontonotes5 tner/wikiann:en tner/btc \
  --split test \
  --max_length 192 \
  --batch_size 4 \
  --per_dataset_eval \
  --save_predictions_jsonl \
  --out_dir outputs/eval_outputs/4ds_test
```

## Inference (text → JSONL)

```bash
PYTHONPATH=src python infer.py \
  --checkpoint models/deberta_fullft_multi_4ds \
  --text "OpenAI is based in San Francisco." \
  --output_jsonl outputs/infer_outputs/predictions.jsonl
```

## Train (optional)

```bash
PYTHONPATH=src python train.py --config configs/train.yaml
```
