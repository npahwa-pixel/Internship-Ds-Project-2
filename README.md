# NER → JSON Extraction (DeBERTa + LoRA)

This project builds an end-to-end **Named Entity Recognition (NER) → JSON** extraction pipeline using **`microsoft/deberta-v3-large`**. The system trains token-classification models, converts BIO tags into entity spans with offsets, and exposes a minimal **Gradio UI** for demo + manual corrections.

This repo contains:
- Training + evaluation scripts under `src/`
- UI under `ui/` and launcher scripts under `scripts/`
- Model checkpoints under `models/` (or downloadable via Drive links)

---

## Demo Video + Model Download Links

### UI Demo Video
```text
https://docs.google.com/document/d/1-8FDQSJbfp_5PFaARHhpRhoY1qvQywEIbfCq5h0xO74/edit?tab=t.0
```
### Model Download Links
```text
https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0
```

## Problem Statement

Given an input sentence, extract named entities and return a stable JSON schema.

**Input**
> John Smith joined OpenAI in San Francisco.

**Output**
```json
{"entities":[
  {"type":"PER","text":"John Smith","start":0,"end":10},
  {"type":"ORG","text":"OpenAI","start":18,"end":24},
  {"type":"LOC","text":"San Francisco","start":28,"end":41}
]}
```
**Why JSON (instead of only BIO tags)?**

- Downstream systems need **spans + offsets + types** in a structured format.

- Makes post-processing, analytics, indexing, and UI display much simpler.

### Architecture

**End-to-End Pipeline**

<img width="4439" height="165" alt="image" src="https://github.com/user-attachments/assets/1d280a24-e1c1-436f-90ef-7d03c42d3c89" />






**Training Variants**






<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8e0b9749-2c84-44a6-b97c-0ca3aa2041eb" />

### Project Structure
Top-level folders in this repo:
- models/ — checkpoints (or downloaded via Drive)

- scripts/ — helper scripts to run evals and UI

- src/ — dataset loaders, preprocessing, training, evaluation, inference

- ui/ — Gradio app

- systemd/ — optional service definition

- requirements_ui.txt — dependencies

- README_UI.md — UI notes

### Pipeline Flow

## 1) Problem Fit & Scope
- **Goal:** High-quality NER extraction and a **JSON-formatted** output for easy consumption.
- **Scope:** Train/evaluate on **CoNLL-2003** and test generalization via additional datasets (**OntoNotes5**, **WikiANN**, **WNUT17**).

## 2) Data Acquisition & Quality
Datasets are loaded via **Hugging Face datasets** (CoNLL-2003 and external corpora supported in `src/`).

**Key correctness checks:**
- Tokenization alignment must use a **fast tokenizer** (`word_ids()` alignment).
- Tag mapping consistency (**dataset tag-space must match model `label2id`** when doing strict evaluation).

## 3) Baseline & Experiments
We evaluate **3 approaches**:
- **Head fine-tune** on CoNLL (`checkpoint-1200`, `checkpoint-1756`)
- **LoRA fine-tune** on CoNLL (best checkpoint shown below)
- **Full fine-tune multi-dataset** (reported in “Extended Experiments”; models linked in Drive)

## 4) Training Correctness & Efficiency
- Head fine-tune updates classifier parameters (and potentially encoder depending on config).
- LoRA fine-tune updates low-rank adapters → faster iteration + smaller artifacts.
- Multi-dataset FullFT improves robustness when training label schema is unified.

## Clone & Setup (Start Here)

> **Start point:** `Final-ds-projects/` is the umbrella directory.  
> Choose which pipeline you want to run:
> - `ner_json_conll_deberta/` (CoNLL Head/LoRA + logs + pipeline UI)
> - `ner_json_deberta_fullft_multi/` (FullFT multi-dataset 3ds)
> - `ner_json_deberta_fullft_multi_4ds/` (FullFT multi-dataset 4ds — best generalization)
> - `unified_ner_ui/` (single UI to demo all pipelines)

### 1) Clone the repository
```bash
git clone <Repository-Here>

```
### 2) Go to the umbrella folder

- ner_json_conll_deberta  
- ner_json_deberta_fullft_multi  
- ner_json_deberta_fullft_multi_4ds  
- unified_ner_ui

### 3) Create & activate a virtual environment (recommended)
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```
### 4) Install dependencies (pick one)

### Option A: CoNLL pipeline deps (Head/LoRA + UI + eval)
```bash
cd ner_json_conll_deberta
pip install -r requirements_ui.txt
```
### Option B: FullFT multi-dataset (3ds)
```bash
cd ner_json_deberta_fullft_multi
pip install -r requirements.txt
```
### Option C: FullFT multi-dataset (4ds)
```bash
cd ner_json_deberta_fullft_multi_4ds
pip install -r requirements.txt
```
### Option D: Unified UI (recommended for demo)
```bash
cd unified_ner_ui
pip install -r requirements.txt
```
### 5) Download model checkpoints (external)
Model files are not stored in GitHub. Download from:
```
https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0
```
Place models into the correct pipeline folder (example):
```
Final-ds-projects/ner_json_conll_deberta/models/...
Final-ds-projects/ner_json_deberta_fullft_multi/models/...
Final-ds-projects/ner_json_deberta_fullft_multi_4ds/models/...
```
### Run UI / Demo
### A) Run unified UI (recommended)
```
cd Final-ds-projects/unified_ner_ui
bash run.sh
```
The launcher runs:
```
python -m ui.gradio_app --host 127.0.0.1 --port 7860 --models_dir models --jsonl_path active_learning/corrections.jsonl
```

### Troubleshooting
```
pip install sentencepiece protobuf
```
### How to Run Evaluations
### 1) CoNLL Head model (CoNLL test)
```
python src/eval_conll_base.py \
  --model_id models/deberta_conll_head/checkpoint-1200 \
  --split test --batch_size 1 --max_length 192 --print_report
```
```
python src/eval_conll_base.py \
  --model_id models/deberta_conll_head/checkpoint-1756 \
  --split test --batch_size 1 --max_length 192 --print_report
```
### 2) CoNLL LoRA model (CoNLL test)

```
python src/eval_conll_lora.py \
  --model_dir models/deberta_conll_lora/checkpoint-1756 \
  --split test --batch_size 1 --max_length 192 --print_report
```
### 3) FullFT (3ds) evaluation
```
PYTHONPATH=src python evaluate.py --config configs/eval.yaml
```
### 4) FullFT (4ds) evaluation
```
PYTHONPATH=src python evaluate.py --config configs/eval.yaml
```

### Evalaution of Multi-dataset Full Fine-tune
In the extended project (models linked above), we trained a FullFT multi-dataset model.
### Per-dataset Test F1 (FullFT example)

| Model | Dataset | Test F1 |
|---|---|---:|
| FullFT (`checkpoint-14319`) | CoNLL2003 | 0.9082 |
| FullFT (`checkpoint-14319`) | OntoNotes5 | 0.9019 |
| FullFT (`checkpoint-14319`) | WNUT2017 | 0.5235 |

**Additional notes:**
- CONCAT test F1 ≈ 0.8833  
- JSON validity on test datasets: 1.0  
- All FullFT models & artifacts are available via the Drive link above.

## Final Model Selection (Best Overall): 4-Dataset Full Fine-Tune — DeBERTa-v3-large

This is the **final selected model** because it was **fully fine-tuned** on **4 datasets** (CoNLL2003 + OntoNotes5 + WikiANN:en + BTC) using a unified NER label space (PER/ORG/LOC/MISC). It provides the most **stable cross-dataset performance** compared to models trained only on CoNLL.

---

### Aggregate Test Metrics (All 4 datasets combined)

```json
{
  "eval_loss": 0.1634780913591385,
  "eval_model_preparation_time": 0.0021,
  "eval_f1": 0.864136846244593,
  "eval_runtime": 428.9256,
  "eval_samples_per_second": 55.289,
  "eval_steps_per_second": 13.823
}
```
## Per-dataset Test F1 (4-Dataset FullFT)

| Dataset | Test F1 |
|---|---:|
| tner/conll2003 | 0.8913 |
| tner/ontonotes5 | 0.8910 |
| tner/wikiann:en | 0.8405 |
| tner/btc | 0.8341 |

- JSON validity on test datasets: 1.0  
- All FullFT models & artifacts are available via the Drive link above.

## Full Per-dataset Metrics Dump
```
  "tner/conll2003": {
    "eval_loss": 0.13971713185310364,
    "eval_model_preparation_time": 0.0021,
    "eval_f1": 0.8913472257953863,
    "eval_runtime": 54.9901,
    "eval_samples_per_second": 62.793,
    "eval_steps_per_second": 15.712
  },
  "tner/ontonotes5": {
    "eval_loss": 0.05927102267742157,
    "eval_model_preparation_time": 0.0021,
    "eval_f1": 0.891015745600494,
    "eval_runtime": 176.6627,
    "eval_samples_per_second": 46.767,
    "eval_steps_per_second": 11.695
  },
  "tner/wikiann:en": {
    "eval_loss": 0.2690074145793915,
    "eval_model_preparation_time": 0.0021,
    "eval_f1": 0.8405108194395176,
    "eval_runtime": 138.4779,
    "eval_samples_per_second": 72.214,
    "eval_steps_per_second": 18.053
  },
  "tner/btc": {
    "eval_loss": 0.11202248185873032,
    "eval_model_preparation_time": 0.0021,
    "eval_f1": 0.8341013824884794,
    "eval_runtime": 49.5973,
    "eval_samples_per_second": 40.325,
    "eval_steps_per_second": 10.081
  }
}
```

### Why This Is the Best Model (Final Reasoning)

✅ **We choose the 4-dataset Full Fine-Tuned model as the final/best model** because:

- **Best cross-dataset stability:** F1 stays consistently high across all datasets (**0.834–0.891**), indicating robust performance beyond a single benchmark.
- **Real-world generalization:** Multi-dataset training reduces overfitting to CoNLL-specific annotation patterns and helps the model adapt to different domains and label distributions.
- **Unified label space:** Avoids tag-space mismatch issues and yields more reliable predictions on unseen corpora.
- **Strong aggregate score:** The combined (aggregate) test F1 is **0.8641**, demonstrating strong overall performance rather than optimizing only one dataset.

**Conclusion:**
- If you only care about the **best CoNLL test F1**, LoRA-on-CoNLL can be the strongest.
- But for a **general NER→JSON model** that behaves well across multiple datasets, the **4-dataset FullFT model is the best final choice**.

## Project Structure

## Project Structure (Umbrella Repository)

This is an **umbrella repository** that contains **multiple pipelines** (CoNLL Head/LoRA, FullFT Multi, FullFT Multi-4DS) plus a unified UI. Each pipeline is self-contained (configs + src + models + outputs).

```text
path/
├── README.md                         # Umbrella README (this file)
├── ner_json_conll_deberta/           # Pipeline 1: CoNLL Head + LoRA + transfer eval
│   ├── models/                       # CoNLL head checkpoints + LoRA adapters
│   ├── reports/                      # eval logs (conll + transfer) saved as .txt
│   ├── scripts/                      # run_all_evals.sh, run_gradio_*.sh, stop scripts
│   ├── src/                          # nerjson_conll package (data/train/eval/infer modules)
│   ├── ui/                           # gradio_app.py (pipeline-specific UI)
│   ├── systemd/                      # optional service unit for running UI
│   ├── requirements_ui.txt           # dependencies for this pipeline + UI
│   └── README_UI.md                  # UI usage notes for this pipeline
│
├── ner_json_deberta_fullft_multi/    # Pipeline 2: FullFT Multi-dataset (3 datasets)
│   ├── configs/                      # train/eval/infer yaml configs
│   ├── models/                       # checkpoint-14319 + eval_outputs (json metrics + jsonl)
│   ├── src/                          # nerjson package (data/train/eval/infer modules)
│   ├── legacy/                       # old monolithic scripts (kept for reference; ignored in git)
│   ├── train.py                      # entrypoint wrapper (calls src/nerjson/*)
│   ├── evaluate.py                   # entrypoint wrapper
│   ├── infer.py                      # entrypoint wrapper
│   ├── requirements.txt              # dependencies for this pipeline
│   └── README.md                     # pipeline-specific README
│
├── ner_json_deberta_fullft_multi_4ds/# Pipeline 3: FullFT Multi-dataset (4 datasets)
│   ├── configs/                      # train/eval/infer yaml configs
│   ├── models/                       # checkpoint-12538 + artifacts
│   ├── outputs/                      # eval_outputs/* (per-dataset + concat metrics JSON)
│   ├── src/                          # nerjson package (data/train/eval/infer modules)
│   ├── legacy/                       # old monolithic scripts (kept for reference; ignored in git)
│   ├── train.py                      # entrypoint wrapper
│   ├── evaluate.py                   # entrypoint wrapper
│   ├── infer.py                      # entrypoint wrapper
│   ├── requirements.txt              # dependencies for this pipeline
│   └── README.md                     # pipeline-specific README
│
└── unified_ner_ui/                   # Single UI for all pipelines (recommended)
    ├── app.py                         # unified Gradio app (model selector → JSON output)
    ├── models.json                    # registry of model paths for all pipelines
    ├── requirements.txt               # UI dependencies
    ├── run.sh                         # starts UI
    └── README.md                      # unified UI instructions
```

### How to use this structure
- For **CoNLL head/LoRA experiments + evaluation logs** → use `ner_json_conll_deberta/`
- For **FullFT multi-dataset (3 datasets)** → use `ner_json_deberta_fullft_multi/`
- For **best generalization (4 datasets FullFT)** → use `ner_json_deberta_fullft_multi_4ds/`
- For a **single demo UI for all models** → use `unified_ner_ui/`

---

### Model downloads (external)
Model checkpoints are stored externally and can be downloaded from:

```text
https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0
```

## Acknowledgements
- **Hugging Face** — `transformers` and `datasets` libraries for model training and dataset loading.
- **Microsoft Research** — DeBERTa model family (`microsoft/deberta-v3-large`).
- **PEFT / LoRA** — parameter-efficient fine-tuning via low-rank adapters.
- **seqeval** — standard NER evaluation metrics (precision/recall/F1).
- **Gradio** — lightweight UI framework for building the demo interface.
