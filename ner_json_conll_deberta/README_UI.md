# NER UI + Active Learning (Gradio)

This patch adds a **browser UI** that lets you:
- Choose between your trained **head** checkpoints and **LoRA** adapters
- Run inference on free-form text
- Output in **JSON / XML / Markdown / Plain**
- Correct predictions and save to **JSONL** for active learning / re-training

## 1) Install deps

```bash
pip install -r requirements_ui.txt
```

## 2) Run UI (foreground)

From repo root:

```bash
python -m ui.gradio_app --host 127.0.0.1 --port 7860 --models_dir models --jsonl_path active_learning/corrections.jsonl
```

Open: http://127.0.0.1:7860

## 3) Run as daemon (nohup)

```bash
bash scripts/run_gradio_daemon.sh
# stop:
bash scripts/stop_gradio.sh
```

Logs: `logs/gradio_7860.log`

## 4) Model discovery

The dropdown auto-scans `models/`:
- LoRA adapters: directories containing `adapter_config.json`
- Full token classification models: directories containing `config.json` and weights (`model.safetensors` or `pytorch_model.bin`)
- Also includes nested `checkpoint-*` directories one level deep.

You can override the dropdown by typing a **custom path / HF id** in the UI.

## 5) JSONL format (active learning)

Each correction appends one line to:

`active_learning/corrections.jsonl`

Example:

```json
{"text":"...","entities":[{"type":"ORG","text":"OpenAI","start":0,"end":6}],
 "meta":{"saved_at":"...","model_key":"head:...","dataset":"conll","split":"validation","source":"gradio_correction_ui"}}
```

You can later convert this JSONL into a training split for re-training / LoRA updates.
