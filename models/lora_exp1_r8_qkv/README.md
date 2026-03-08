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

# Model Card — `lora_exp1_r8_qkv` (DeBERTa-v3-large LoRA Adapter)

This folder contains a **LoRA adapter** fine-tuned for **CoNLL-2003 NER** on top of **DeBERTa-v3-large**.  
It’s designed to be lightweight and easy to ship while still providing strong NER performance.

---

## Model Details

### Model Description

- **Developed by:** Naman Pahwa (npahwa-pixel)
- **Model type:** DeBERTa-v3-large + LoRA adapter (PEFT) for token classification (NER)
- **Base model:** `tner/deberta-v3-large-conll2003`
- **Adapter name:** `lora_exp1_r8_qkv`
- **LoRA target:** QKV projection modules (attention projections)
- **LoRA rank:** 8 (R=8)
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

Best when you want:
- a **small adapter artifact** instead of full fine-tuned weights
- faster fine-tuning iteration (parameter-efficient)
- strong CoNLL-2003 style NER

### Downstream Use

Common integrations:
- NER → JSON extraction (entity spans + offsets)
- document tagging & indexing
- UI highlighting and analytics

### Out-of-Scope Use

- Non-English NER without retraining
- Domain-specific schemas (medical/legal) without adaptation
- Cross-dataset strict evaluation when tag-space differs (requires mapping)

---

## Bias, Risks, and Limitations

- May inherit dataset/model biases from CoNLL-2003 + base model.
- Performance may drop in domains far from newswire.
- For datasets with different label spaces, strict evaluation may not apply directly.

### Recommendations

- For cross-dataset evaluation, use coarse transfer mapping (PER/ORG/LOC/MISC).
- Validate outputs in downstream systems.

---

## How to Get Started

### 1) Install dependencies

```bash
pip install -r ../../requirements_ui.txt
