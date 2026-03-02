#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/data1/tanvekar/MAYO_AIM2"
EVAL_PY="${PROJECT_DIR}/eval_run_baselines_v2.py"
GT_PATH="${GT_PATH:-${PROJECT_DIR}/data/natural_question_1500.json}"
DB_PATH="${DB_PATH:-${PROJECT_DIR}/data/database.db}"
COMPUTE_BERTSCORE="${COMPUTE_BERTSCORE:-0}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/results/eval_baselines_v2}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

declare -A MODEL_LOG_ROOTS=(
  ["gemma27b"]="${PROJECT_DIR}/logs/Gemma/27B/sql_with_conf"
  ["qwen3_30b_a3b"]="${PROJECT_DIR}/logs/Qwen3/30B-A3B/sql_with_conf"
)

STYLES=("zero_shot" "few_shot" "cot")

latest_pred_for_style() {
  local model_root="$1"
  local style="$2"
  local latest_dir=""
  latest_dir="$(ls -dt "${model_root}/${style}_"* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_dir}" ]]; then
    echo ""
    return 0
  fi
  if [[ -f "${latest_dir}/generated_sql.jsonl" ]]; then
    echo "${latest_dir}/generated_sql.jsonl"
    return 0
  fi
  echo ""
}

echo "Eval output dir: ${OUT_DIR}"
echo

for model_key in "${!MODEL_LOG_ROOTS[@]}"; do
  model_root="${MODEL_LOG_ROOTS[$model_key]}"
  echo "== Model: ${model_key}"
  echo "   Logs:  ${model_root}"

  for style in "${STYLES[@]}"; do
    pred_path="$(latest_pred_for_style "${model_root}" "${style}")"
    if [[ -z "${pred_path}" ]]; then
      echo "  - ${style}: skipped (no run directory found)"
      continue
    fi

    out_json="${OUT_DIR}/eval_${model_key}_${style}.json"
    echo "  - ${style}: evaluating"
    echo "      pred: ${pred_path}"
    echo "      out : ${out_json}"

    python "${EVAL_PY}" \
      --pred_path "${pred_path}" \
      --gt_path "${GT_PATH}" \
      --db_path "${DB_PATH}" \
      --output_json "${out_json}" \
      --compute_bertscore "${COMPUTE_BERTSCORE}"
  done
  echo
done

echo "Done. All eval outputs are in: ${OUT_DIR}"
