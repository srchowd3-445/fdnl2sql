#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data1/srchowd3/MAYO_AIM2
if [[ -f .env ]]; then set -a; source .env; set +a; fi
LOG_FILE=/mnt/data1/srchowd3/MAYO_AIM2/logs/decompose_batch/decomp_final9_8192.log
echo "[$(date +"%Y-%m-%d %H:%M:%S")] starting batch run" | tee -a "$LOG_FILE"
if python /mnt/data1/srchowd3/MAYO_AIM2/decompose_natural_questions_batch.py --mode batch_run --input_json /mnt/data1/srchowd3/MAYO_AIM2/data/natural_question_additional_200_naturalized.json --schema_json /mnt/data1/srchowd3/MAYO_AIM2/data/schema.json --output_json /mnt/data1/srchowd3/MAYO_AIM2/data/natural_question_additional_200_decomposed.json --batch_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_input.jsonl --batch_meta_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_meta.json --batch_output_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_output.jsonl --batch_error_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_error.jsonl --start 0 --limit -1 --input_json /mnt/data1/srchowd3/MAYO_AIM2/data/natural_question_additional_200_decomposed_final9_input.json --start 0 --limit -1 --max_tokens 8192 --token_param max_completion_tokens --output_json /mnt/data1/srchowd3/MAYO_AIM2/data/natural_question_additional_200_decomposed_final9_8192.json --batch_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_input_final9_8192.jsonl --batch_meta_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_meta_final9_8192.json --batch_output_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_output_final9_8192.jsonl --batch_error_jsonl_path /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_error_final9_8192.jsonl --raw_output_jsonl /mnt/data1/srchowd3/MAYO_AIM2/data/decompose_batch_raw_final9_8192.jsonl 2>&1 | tee -a "$LOG_FILE"; then
  rc=0
else
  rc=$?
fi
echo "[$(date +"%Y-%m-%d %H:%M:%S")] finished rc=$rc" | tee -a "$LOG_FILE"
exit $rc
