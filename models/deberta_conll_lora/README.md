---
base_model: tner/deberta-v3-large-conll2003
library_name: peft
tags:
- base_model:adapter:tner/deberta-v3-large-conll2003
- lora
- transformers
- token-classification
- ner
- conll2003
---

# Model Card — `deberta_conll_lora` (DeBERTa-v3-large LoRA Adapter)

This folder contains a **LoRA adapter** fine-tuned on **CoNLL-2003 NER** on top of **DeBERTa-v3-large**.  
It is a parameter-efficient alternative to full fine-tuning: small artifacts, fast iteration, and easy distribution.

---

## Model Details

### Model Description

- **Developed by:** Naman Pahwa (npahwa-pixel)
- **Model type:** DeBERTa-v3-large + LoRA adapter (PEFT) for token classification (NER)
- **Base model:** `tner/deberta-v3-large-conll2003`
- **Adapter name:** `deberta_conll_lora`
- **Training objective:** Named Entity Recognition (token classification)
- **Language(s):** English
- **Label set:** CoNLL-2003 BIO tags (PER / ORG / LOC / MISC)
- **License:** Follow the base model license / research use constraints

### Model Sources

- **Project repository:** `Internship-Ds-Project-2` (this repo)
- **Base model:** `tner/deberta-v3-large-conll2003` (Hugging Face)

---

## Uses

### Direct Use

Use this adapter for **English NER** in the CoNLL-2003 schema:
- PERSON (PER)
- ORGANIZATION (ORG)
- LOCATION (LOC)
- MISC (MISC)

Typical use cases:
- entity extraction + highlighting
- NER → JSON pipelines (spans + offsets + types)
- indexing / analytics / search enrichment

### Downstream Use

- Plug into a UI (Gradio) that outputs JSON entities
- Run batch inference on text corpora
- Evaluate generalization using coarse mapping across datasets (PER/ORG/LOC/MISC)

### Out-of-Scope Use

- Non-English NER without adaptation
- Strict evaluation on datasets with different tag-space unless you map labels
- Domain-specific schemas (biomedical/legal) without retraining

---

## Bias, Risks, and Limitations

- Inherits biases from the base model and the CoNLL-2003 dataset.
- Can underperform on informal / social text, unless adapted further.
- Cross-dataset strict evaluation may fail if label/tag spaces differ.

### Recommendations

- For cross-dataset comparisons, use a **coarse transfer mapping** (PER/ORG/LOC/MISC).
- Validate entity spans/offsets in downstream production use.

---

## How to Get Started

### 1) Install dependencies

From the parent pipeline folder `ner_json_conll_deberta/`:

```bash
pip install -r requirements_ui.txt
