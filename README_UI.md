# README — UI (Unified Gradio Demo)

This document explains how to run the **Gradio UI** included in this repository to test **NER → JSON** inference interactively.

The UI lets you:
- select a model/run from a dropdown
- type input text
- run inference
- view extracted entities in a stable JSON schema
- optionally store “corrections” (active-learning style) into a JSONL file

---

## Where is the UI?

There are two UI entrypoints in this repo:

1) **Unified UI (recommended for demo)**  
📁 `unified_ner_ui/`

2) **CoNLL pipeline UI (tied to the CoNLL project structure)**  
📁 `ner_json_conll_deberta/ui/`

If you want one place that can demo **all pipelines**, use **unified_ner_ui**.

---

## Demo Video

A short demo video walkthrough is here (Google Doc link):
- https://docs.google.com/document/d/1-8FDQSJbfp_5PFaARHhpRhoY1qvQywEIbfCq5h0xO74/edit?tab=t.0

---

## Model Downloads (External)

Model files are not committed to GitHub. Download them from:
- https://docs.google.com/document/d/1b3ftGIiYNDvyCPWJsa6qJTdI9gkJHAWej7pLpZL7u-8/edit?tab=t.0

After downloading, place them into the correct folder (examples):

```text
ner_json_conll_deberta/models/...
ner_json_deberta_fullft_multi/models/...
ner_json_deberta_fullft_multi_4ds/models/...
```

## Quick Start (Recommended): Unified UI
1) Go to UI folder
   ```cd unified_ner_ui```
   
2) Create venv + install deps
   ```python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Run UI
   ```
   bash run.sh
   ```
   
If you prefer running directly:
   ```
   python app.py
   ```

