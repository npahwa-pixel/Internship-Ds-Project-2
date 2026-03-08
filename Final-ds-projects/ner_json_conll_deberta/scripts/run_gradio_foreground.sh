#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash scripts/run_gradio_foreground.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
MODELS_DIR="${MODELS_DIR:-models}"
JSONL_PATH="${JSONL_PATH:-active_learning/corrections.jsonl}"

python -m ui.gradio_app --host "$HOST" --port "$PORT" --models_dir "$MODELS_DIR" --jsonl_path "$JSONL_PATH"
