#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${MODEL_BASELINE_LOG_DIR:-${BASE_DIR}/results/logs}"
if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
  LOG_DIR="/tmp/mayo_aim2_model_baseline_logs"
  mkdir -p "${LOG_DIR}"
fi
TTY_PATH="$(tty 2>/dev/null || true)"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [model_baseline.py args]"
  echo
  echo "Example:"
  echo "  $0 \\"
  echo "    --input_json ${BASE_DIR}/results/empty_gt_fixed_v7_smoke_exec_eval.json \\"
  echo "    --db_path ${BASE_DIR}/data/database.db \\"
  echo "    --output_json ${BASE_DIR}/results/model_baseline_gcp.json \\"
  echo "    --checkpoint_json ${BASE_DIR}/results/model_baseline_gcp.json.checkpoint.json \\"
  echo "    --backend gcp \\"
  echo "    --gcp_project praxis-flight-482822-q2 \\"
  echo "    --vertex_location us-central1 \\"
  echo "    --gcp_model vertex_ai/gemini-2.5-flash \\"
  echo "    --resume 1 --save_every 1 --print_per_item 1 --print_model_check 1"
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/model_baseline_${ts}.log"
PID_FILE="${LOG_DIR}/model_baseline_${ts}.pid"

nohup python3 "${BASE_DIR}/model_baseline.py" "$@" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

# Completion notifier:
# - Writes a done line to log
# - Prints done message to the same terminal if writable
# - Sends desktop notification if notify-send exists
(
  while kill -0 "${PID}" 2>/dev/null; do
    sleep 2
  done
  DONE_TS="$(date '+%Y-%m-%d %H:%M:%S')"
  MSG="model_baseline.py finished at ${DONE_TS} (pid=${PID})"
  echo "[DONE] ${MSG}" >> "${LOG_FILE}"
  if [ -n "${TTY_PATH}" ] && [ -w "${TTY_PATH}" ]; then
    printf '\n[DONE] %s\nLog: %s\n\n' "${MSG}" "${LOG_FILE}" > "${TTY_PATH}" || true
  fi
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "model_baseline done" "${MSG}" >/dev/null 2>&1 || true
  fi
) >/dev/null 2>&1 &

echo "Started model_baseline.py in background"
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop: kill ${PID}"
