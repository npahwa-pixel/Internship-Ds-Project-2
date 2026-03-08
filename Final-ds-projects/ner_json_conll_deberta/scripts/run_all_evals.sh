#!/usr/bin/env bash
set -u -o pipefail
shopt -s nullglob

# Prefer local venv python if present, else fallback
if [[ -x ".venv/bin/python" ]]; then
  PY="$(pwd)/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi

echo "Using python: $PY"
"$PY" -V

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

mkdir -p reports/transfer/{ontonotes5,wikiann_en,wnut17}

HEAD_CKPTS=(models/deberta_conll_head/checkpoint-*)
LORA_DIRS=(models/deberta_conll_lora models/lora_exp1_r8_qkv models/lora_exp2_r16_qkv)

DATASETS=(
  "ontonotes5:ontonotes5:"
  "wikiann_en:wikiann:--lang en"
  "wnut17:wnut17:"
)

run_transfer () {
  local out_folder="$1"
  local tag="$2"
  shift 2
  local log="reports/transfer/${out_folder}/${tag}.txt"

  echo "==== ${out_folder} | ${tag} ===="
  {
    echo "CMD: $PY src/eval_transfer_coarse.py $*"
    echo "PWD: $(pwd)"
    echo "DATE: $(date)"
    echo "----------------------------------------"
    "$PY" src/eval_transfer_coarse.py "$@"
    echo "----------------------------------------"
    echo "STATUS: OK"
  } 2>&1 | tee "$log" || true

  return 0
}

# HEAD checkpoints (CoNLL-trained)
for model_dir in "${HEAD_CKPTS[@]}"; do
  ckpt="$(basename "$model_dir")"
  for spec in "${DATASETS[@]}"; do
    IFS=":" read -r folder ds extra <<<"$spec"
    run_transfer "$folder" "head_${ckpt}_${ds}_test" \
      --dataset "$ds" $extra --split test \
      --model_dir "$model_dir" \
      --batch_size 4 --max_length 192 --print_report
  done
done

# LoRA checkpoints (CoNLL-trained adapters)
for d in "${LORA_DIRS[@]}"; do
  parent="$(basename "$d")"
  for lora_ckpt in "$d"/checkpoint-*; do
    ckpt="$(basename "$lora_ckpt")"
    for spec in "${DATASETS[@]}"; do
      IFS=":" read -r folder ds extra <<<"$spec"
      run_transfer "$folder" "lora_${parent}_${ckpt}_${ds}_test" \
        --dataset "$ds" $extra --split test \
        --base_model microsoft/deberta-v3-large \
        --lora_dir "$lora_ckpt" \
        --batch_size 4 --max_length 192 --print_report
    done
  done
done

echo "✅ DONE. Transfer logs saved under reports/transfer/{ontonotes5,wikiann_en,wnut17}/"
