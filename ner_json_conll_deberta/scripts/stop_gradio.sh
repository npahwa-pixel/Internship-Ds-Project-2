#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7860}"
PID_FILE="logs/gradio_${PORT}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No pid file at $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "✅ Stopped Gradio pid=$PID"
else
  echo "Process not running (pid=$PID)."
fi
rm -f "$PID_FILE"
