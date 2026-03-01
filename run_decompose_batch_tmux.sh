#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="decompose_batch_${TS}"

if [[ "${1:-}" == "--session" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--session requires a value" >&2
    exit 1
  fi
  SESSION_NAME="$2"
  shift 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run_decompose_batch_tmux.sh [--session SESSION_NAME] [extra args passed to decompose script]

Examples:
  ./run_decompose_batch_tmux.sh
  ./run_decompose_batch_tmux.sh --session decomp200
  ./run_decompose_batch_tmux.sh --session decomp200 --model gpt-5-mini --start 0 --limit 200

Behavior:
- Starts a detached tmux session.
- Loads .env from project root if present.
- Runs decompose_natural_questions_batch.py in batch_run mode.
- Writes logs to logs/decompose_batch/<session>.log
- You can safely close the terminal.
EOF
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists. Use a different --session name." >&2
  exit 1
fi

LOG_DIR="$ROOT_DIR/logs/decompose_batch"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
RUN_FILE="$LOG_DIR/${SESSION_NAME}.cmd.sh"

PY_ARGS=(
  --mode batch_run
  --input_json "$ROOT_DIR/data/natural_question_additional_200_naturalized.json"
  --schema_json "$ROOT_DIR/data/schema.json"
  --output_json "$ROOT_DIR/data/natural_question_additional_200_decomposed.json"
  --batch_jsonl_path "$ROOT_DIR/data/decompose_batch_input.jsonl"
  --batch_meta_path "$ROOT_DIR/data/decompose_batch_meta.json"
  --batch_output_jsonl_path "$ROOT_DIR/data/decompose_batch_output.jsonl"
  --batch_error_jsonl_path "$ROOT_DIR/data/decompose_batch_error.jsonl"
  --start 0
  --limit -1
)

if [[ "$#" -gt 0 ]]; then
  PY_ARGS+=("$@")
fi

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$ROOT_DIR"
  echo 'if [[ -f .env ]]; then set -a; source .env; set +a; fi'
  printf 'LOG_FILE=%q\n' "$LOG_FILE"
  echo 'echo "[$(date +"%Y-%m-%d %H:%M:%S")] starting batch run" | tee -a "$LOG_FILE"'
  printf 'if python %q' "$ROOT_DIR/decompose_natural_questions_batch.py"
  for a in "${PY_ARGS[@]}"; do
    printf ' %q' "$a"
  done
  echo ' 2>&1 | tee -a "$LOG_FILE"; then'
  echo '  rc=0'
  echo 'else'
  echo '  rc=$?'
  echo 'fi'
  echo 'echo "[$(date +"%Y-%m-%d %H:%M:%S")] finished rc=$rc" | tee -a "$LOG_FILE"'
  echo 'exit $rc'
} > "$RUN_FILE"

chmod +x "$RUN_FILE"

tmux new-session -d -s "$SESSION_NAME" "bash $(printf '%q' "$RUN_FILE")"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach:   tmux attach -t $SESSION_NAME"
echo "Detach:   Ctrl+b then d"
echo "Status:   tmux list-sessions | rg $SESSION_NAME"
echo "Tail log: tail -f $LOG_FILE"
