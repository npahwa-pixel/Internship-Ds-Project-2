#!/usr/bin/env bash
set -euo pipefail

# Starts Gradio in the background (daemon-like) using nohup.
#
# Usage (from repo root):
#   bash scripts/run_gradio_daemon.sh
#
# Then open: http://localhost:7860

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
MODELS_DIR="${MODELS_DIR:-models}"
JSONL_PATH="${JSONL_PATH:-active_learning/corrections.jsonl}"

mkdir -p logs
LOG_FILE="logs/gradio_${PORT}.log"
PID_FILE="logs/gradio_${PORT}.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "Already running (pid=$(cat "$PID_FILE")). Stop it first: bash scripts/stop_gradio.sh"
  exit 0
fi

nohup python -m ui.gradio_app --host "$HOST" --port "$PORT" --models_dir "$MODELS_DIR" --jsonl_path "$JSONL_PATH" \
  >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "✅ Started Gradio pid=$! (log=$LOG_FILE)"
