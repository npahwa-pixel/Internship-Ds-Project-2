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

# Model Card — `lora_exp2_r16_qkv` (DeBERTa-v3-large LoRA Adapter)

This repository folder contains a **LoRA adapter** trained on **CoNLL-2003 NER** using **DeBERTa-v3-large**.  
It is intended to be used as a lightweight adapter on top of the base model, enabling high-quality NER predictions while keeping artifacts small compared to a full fine-tuned checkpoint.

---

## Model Details

### Model Description

- **Developed by:** Naman Pahwa (npahwa-pixel)
- **Model type:** DeBERTa-v3-large + LoRA adapter (PEFT) for token classification (NER)
- **Base model:** `tner/deberta-v3-large-conll2003` (DeBERTa-v3-large trained for CoNLL-2003 style NER)
- **Adapter name:** `lora_exp2_r16_qkv`
- **LoRA target:** QKV projection modules (attention projections)  
- **LoRA rank:** 16 (R=16)
- **Language(s):** English
- **Task:** Named Entity Recognition (NER)
- **Labels:** CoNLL-2003 BIO tag set (PER / ORG / LOC / MISC)
- **License:** Same as base model / research use (follow upstream license)

### Model Sources

- **Project repository:** `Internship-Ds-Project-2` (this repo)
- **Base model:** Hugging Face model `tner/deberta-v3-large-conll2003`

---

## Uses

### Direct Use

Use this adapter for **NER token classification** on English text following CoNLL-style entities:
- PERSON (PER)
- ORGANIZATION (ORG)
- LOCATION (LOC)
- MISC (MISC)

This adapter is best used when you want:
- a **smaller artifact** than a full fine-tuned model
- a **fast iteration** fine-tuning approach (adapter-based)
- strong CoNLL-2003 performance

### Downstream Use

This adapter can be plugged into larger systems such as:
- NER → JSON pipelines (entity spans + offsets)
- search indexing / knowledge extraction
- downstream analytics and UI highlighting

### Out-of-Scope Use

- Non-English NER without retraining
- Domain-specific entity schemas (medical/legal) unless adapted
- Cross-dataset strict evaluation when label/tag space differs (requires mapping)

---

## Bias, Risks, and Limitations

- The model reflects biases and limitations of the training data and base model.
- Performance may degrade on text domains that differ substantially from newswire (CoNLL-2003).
- Strict evaluation may fail on datasets with different label/tag spaces unless a mapping is applied.

### Recommendations

- For cross-dataset generalization tests, use a coarse transfer mapping (PER/ORG/LOC/MISC).
- Validate outputs in downstream applications (especially for high-stakes use-cases).

---

## How to Get Started

### 1) Install dependencies

```bash
pip install -r ../../requirements_ui.txt
