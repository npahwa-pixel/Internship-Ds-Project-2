# DeBERTa NER→JSON — CoNLL Head + LoRA (ner_json_conll_deberta)

This folder contains the **CoNLL-2003 pipeline** for **DeBERTa-v3-large** with:
- **Head fine-tuning** checkpoints
- **LoRA fine-tuning** checkpoints
- **Evaluation scripts** (CoNLL + cross-dataset helpers)
- **Gradio UI** for demo/inference
- **NER → JSON output** (type + span text + character offsets)

---

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
**Model Downloads (External)**
Model checkpoints are not stored in GitHub. Download them from:
```
https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0
```
Place them here:
```
ner_json_conll_deberta/
└── models/
    ├── deberta_conll_head/
    ├── deberta_conll_lora/
    ├── lora_exp1_r8_qkv/
    └── lora_exp2_r16_qkv/
```
**Install**
```
cd ner_json_conll_deberta
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ui.txt
```
If you hit missing deps:
```
python -m pip install sentencepiece protobuf
```
**Evaluate (CoNLL)**
**Head (checkpoint-1200)**
```
python src/nerjson_conll/evaluation/eval_conll_base.py \
  --model_id models/deberta_conll_head/checkpoint-1200 \
  --split test --batch_size 1 --max_length 192 --print_report
```
**Head (checkpoint-1756)**
```
python src/nerjson_conll/evaluation/eval_conll_base.py \
  --model_id models/deberta_conll_head/checkpoint-1756 \
  --split test --batch_size 1 --max_length 192 --print_report
```
**LoRA (best: checkpoint-1756)**
```
python src/nerjson_conll/evaluation/eval_conll_lora.py \
  --model_dir models/deberta_conll_lora/checkpoint-1756 \
  --split test --batch_size 1 --max_length 192 --print_report
```
**Project Structure**
```
ner_json_conll_deberta/
├── src/                     # Core ML pipeline (data, preprocessing, training, eval, inference)
│   └── nerjson_conll/
├── ui/                      # Gradio app (model selector → inference → JSON output)
├── scripts/                 # Run/stop UI, run eval helpers
├── reports/                 # Evaluation logs (per dataset) + transfer logs
├── systemd/                 # Optional systemd unit for Gradio
├── models/                  # External checkpoints (download link above; ignored in git)
├── eval_conll.py            # Wrapper entrypoint
├── eval_conll_lora.py       # Wrapper entrypoint
├── eval_transfer.py         # Wrapper entrypoint
├── README_UI.md             # UI usage notes
└── requirements_ui.txt      # Dependencies for this pipeline
```
**Acknowledgements**
- Hugging Face: transformers, datasets

- Microsoft DeBERTa: microsoft/deberta-v3-large

- PEFT / LoRA: parameter-efficient adapters

- ```seqeval```: NER scoring

- Gradio: demo UI
