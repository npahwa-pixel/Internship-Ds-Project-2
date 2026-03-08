# Unified NER → JSON Gradio UI

This UI runs inference for your three projects inside `Final-ds-projects`:

1. CoNLL DeBERTa head (best checkpoint under `.../models/deberta_conll_head/`)
2. Full fine-tune multi-dataset (checkpoint-14319)
3. Full fine-tune 4-dataset (checkpoint-12538)

## Install + Run

```bash
cd /Users/path

python -m venv .venv_ui
source .venv_ui/bin/activate
pip install -r unified_ner_ui/requirements.txt

export PROJECT_ROOT="/Users/path"
python unified_ner_ui/app.py
```

## Custom model path

Use the Advanced box to point to any HF checkpoint directory:
- must contain `config.json` + `model.safetensors` (or `pytorch_model.bin`)
- LoRA adapter dirs also work if they contain `adapter_config.json` + `adapter_model.safetensors` (requires `peft`)

## Optional flattening (not required)

If you want to simplify paths, you can create symlinks and edit `models.json`:

```bash
cd /Users/path
ln -s ner_json_conll_deberta/ner_json_conll_deberta conll_project
ln -s ner_json_deberta_fullft_multi/ner_json_deberta_fullft_multi fullft_multi_project
ln -s ner_json_deberta_fullft_multi_4ds/ner_json_deberta_fullft_multi_4ds fullft_4ds_project
```
