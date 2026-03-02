#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data2/srchowd3/miniconda3/envs/myenv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared/shared_hf_home/hub/models--MPX0222forHF--SQL-R1-7B/snapshots/db409e8372ca5e463126b07e905b5245caf14ea6}"

INPUT_JSON="${INPUT_JSON:-$ROOT_DIR/data/natural_question_1500.json}"
SCHEMA_JSON="${SCHEMA_JSON:-$ROOT_DIR/data/schema.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT_DIR/results/nl-sql-r1-smoke.json}"

START="${START:-0}"
LIMIT="${LIMIT:-10}"   # set LIMIT=-1 to run all questions
GPU="${GPU:-6}"

LOG_DIR="$ROOT_DIR/results/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/nl-sql-r1-smoke_${TS}.log"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/run_sqlr1_smoke.py"
  --model_path "$MODEL_PATH"
  --input_json "$INPUT_JSON"
  --schema_json "$SCHEMA_JSON"
  --output_json "$OUTPUT_JSON"
  --start "$START"
  --limit "$LIMIT"
  --gpu "$GPU"
  --dtype bfloat16
  --tensor_parallel_size 1
  --gpu_memory_utilization 0.92
  --max_model_len 8192
  --max_new_tokens 1024
  --temperature 0.0
  --top_p 1.0
  --batch_size 4
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started background SQL-R1 smoke run"
echo "PID:      $PID"
echo "Log file: $LOG_FILE"
echo "Output:   $OUTPUT_JSON"
echo "Watch:    tail -f $LOG_FILE"
