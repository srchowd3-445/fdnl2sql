#!/bin/bash

# =====================================================
# LOCAL GPU SELECTION
# =====================================================

if [ ! -z "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
TP_SIZE=$NUM_GPUS

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
GPU_MEM=0.85

VLLM_PORT=8000
VLLM_HOST="127.0.0.1"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

PROJECT_DIR="/mnt/data1/tanvekar/MAYO_AIM2"
LOG_ROOT="${PROJECT_DIR}/logs/Qwen3/30B-A3B/sql_with_conf"
mkdir -p "$LOG_ROOT"

RUN_ID=$(date +%Y%m%d_%H%M%S)

PROMPT_STYLE="${PROMPT_STYLE:-cot}"
INPUT_PATH="${INPUT_PATH:-${PROJECT_DIR}/data/natural_question_1500.json}"
ZS_PROMPT_FILE="${ZS_PROMPT_FILE:-${PROJECT_DIR}/prompting/zero_shot_sql_expert.txt}"
FEW_SHOT_FILE="${FEW_SHOT_FILE:-${PROJECT_DIR}/prompting/few_shot_sql.txt}"
COT_PROMPT_FILE="${COT_PROMPT_FILE:-${PROJECT_DIR}/prompting/cot.txt}"

LOGPROB_MODE="${LOGPROB_MODE:-structured}"
USE_PYDANTIC="${USE_PYDANTIC:-1}"

RUN_LOG_DIR="${LOG_ROOT}/${PROMPT_STYLE}_${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Starting Qwen3-30B SQL Baseline"
echo "Prompt style: $PROMPT_STYLE"
echo "GPU ids: $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Run ID: $RUN_ID"
echo "Pydantic schema: $USE_PYDANTIC"
echo "Logprob mode: $LOGPROB_MODE"
echo "=========================================="

# =====================================================
# SERVER HEALTH CHECK
# =====================================================

check_server_ready() {
    for i in {1..60}; do
        if curl -s -f "${VLLM_API_BASE%/v1}/health" > /dev/null 2>&1; then
            echo "✅ vLLM server ready"
            return 0
        fi
        echo "Waiting for server... ($i/60)"
        sleep 5
    done
    echo "❌ Server failed to start"
    return 1
}

cleanup() {
    echo ""
    echo "Stopping vLLM..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
    fi
}

trap cleanup EXIT INT TERM

# =====================================================
# START vLLM
# =====================================================

echo "Launching vLLM..."

vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TP_SIZE \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization $GPU_MEM \
    --chat-template-content-format string \
    > "${RUN_LOG_DIR}/vllm.log" 2>&1 &

VLLM_PID=$!

if ! check_server_ready; then
    echo "Check log: ${RUN_LOG_DIR}/vllm.log"
    exit 1
fi

# =====================================================
# RUN INFERENCE
# =====================================================

python run_baselines.py \
    --api_base "$VLLM_API_BASE" \
    --api_key "dummy" \
    --model "$MODEL_NAME" \
    --input_path "$INPUT_PATH" \
    --input_format auto \
    --output_path "${RUN_LOG_DIR}/generated_sql.jsonl" \
    --error_path "${RUN_LOG_DIR}/errors.jsonl" \
    --checkpoint_path "${RUN_LOG_DIR}/checkpoint.json" \
    --schema_file "${PROJECT_DIR}/data/schema.json" \
    --db_path "${PROJECT_DIR}/data/database.db" \
    --question_keys "natural_question" \
    --prompt_style "$PROMPT_STYLE" \
    --zs_prompt_file "$ZS_PROMPT_FILE" \
    --few_shot_file "$FEW_SHOT_FILE" \
    --cot_prompt_file "$COT_PROMPT_FILE" \
    --resume 1 \
    --save_every 50 \
    --batch_size 128 \
    --batch_concurrency 16 \
    --temperature 0.0 \
    --max_tokens 512 \
    --timeout 120 \
    --num_retries 2 \
    --use_pydantic_schema "$USE_PYDANTIC" \
    --logprob_mode "$LOGPROB_MODE"

echo "=========================================="
echo "Run complete."
echo "Outputs: $RUN_LOG_DIR"
echo "=========================================="