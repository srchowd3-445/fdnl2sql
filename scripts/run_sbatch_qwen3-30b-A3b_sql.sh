#!/bin/bash
#SBATCH --job-name=gemma27b-baselines
#SBATCH --account=3dllms
#SBATCH --partition=inv-ssheshap
#SBATCH --gres=gpu:2
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=/gscratch/tkishore/MAYO_AIM2/logs/gemma27b_baselines_%j.out
#SBATCH --error=/gscratch/tkishore/MAYO_AIM2/logs/gemma27b_baselines_%j.err

# ==========================================
# CONFIG
# ==========================================

MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
TP_SIZE=2
GPU_MEM=0.85

VLLM_PORT=8000
VLLM_HOST="127.0.0.1"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

PROJECT_DIR="/gscratch/tkishore/MAYO_AIM2/"
LOG_DIR="${PROJECT_DIR}/logs/Qwen3/30B-A3B/sql_with_conf_ZS"
mkdir -p "$LOG_DIR"

PROMPT_STYLE="${PROMPT_STYLE:-zero-shot}"            # zero_shot | few_shot | cot
PROMPT_STYLES="${PROMPT_STYLES:-$PROMPT_STYLE}"     # comma-separated, e.g. zero_shot,few_shot
INPUT_PATH="${INPUT_PATH:-${PROJECT_DIR}/data/natural_question_1500.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${LOG_DIR}/generated_sql_${PROMPT_STYLE}.jsonl}"
ERROR_FILE="${ERROR_FILE:-${LOG_DIR}/error_log_${PROMPT_STYLE}.jsonl}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${OUTPUT_FILE}.checkpoint.json}"
OUTPUT_FILE_TEMPLATE="${OUTPUT_FILE_TEMPLATE:-${LOG_DIR}/generated_sql_{style}.jsonl}"
ERROR_FILE_TEMPLATE="${ERROR_FILE_TEMPLATE:-${LOG_DIR}/error_log_{style}.jsonl}"
CHECKPOINT_FILE_TEMPLATE="${CHECKPOINT_FILE_TEMPLATE:-${LOG_DIR}/generated_sql_{style}.jsonl.checkpoint.json}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PROJECT_DIR}/results/logs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_CONCURRENCY="${BATCH_CONCURRENCY:-16}"
LOGPROB_MODE="${LOGPROB_MODE:-structured}"           # structured | none
ZS_PROMPT_FILE="${ZS_PROMPT_FILE:-${PROJECT_DIR}/prompting/zero_shot_sql_expert.txt}"
FEW_SHOT_FILE="${FEW_SHOT_FILE:-${PROJECT_DIR}/prompting/few_shot_sql.txt}"

# ==========================================
# ENVIRONMENT SETUP
# ==========================================

module --force purge
module load arcc/1.0
module load gcc/14.2.0
module load miniconda3
module load cuda-toolkit/12.8.0

source activate /gscratch/tkishore/vllm

export XDG_CACHE_HOME=/gscratch/tkishore/cache
export HF_HOME=/gscratch/tkishore/cache/hf_cache
export HF_HUB_CACHE=/gscratch/tkishore/cache/hf_cache/hub
export TORCH_HOME=/gscratch/tkishore/cache/torch
export FLASHINFER_CACHE_DIR=/gscratch/tkishore/cache/flashinfer

cd $PROJECT_DIR

echo "=========================================="
echo "Starting Gemma-3-27B Baseline SQL Job"
echo "Prompt style: $PROMPT_STYLE"
echo "Prompt styles: $PROMPT_STYLES"
echo "Input path: $INPUT_PATH"
echo "Output file: $OUTPUT_FILE"
echo "Zero-shot prompt: $ZS_PROMPT_FILE"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TP_SIZE"
echo "GPU Memory Utilization: $GPU_MEM"
echo "Port: $VLLM_PORT"
echo "=========================================="

# ==========================================
# SERVER HEALTH CHECK
# ==========================================

check_server_ready() {
    local max_attempts=60
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "${VLLM_API_BASE%/v1}/health" > /dev/null 2>&1; then
            echo "✅ vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
            echo "Waiting for vLLM server... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "❌ vLLM server failed to start"
    return 1
}

cleanup() {
    echo ""
    echo "Cleaning up vLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        echo "✅ vLLM server stopped"
    fi
}

trap cleanup EXIT INT TERM

# ==========================================
# START vLLM SERVER
# ==========================================

echo ""
echo "Launching vLLM server..."

vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $TP_SIZE \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization $GPU_MEM \
    --chat-template-content-format openai \
    > "${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server
if ! check_server_ready; then
    echo "❌ Server failed to start. Check log:"
    echo "${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log"
    exit 1
fi

# ==========================================
# RUN BASELINES
# ==========================================

echo ""
echo "Running SQL baseline inference..."
echo "Prompt styles to run: $PROMPT_STYLES"
echo ""

IFS=',' read -r -a STYLE_ARR <<< "$PROMPT_STYLES"
NUM_STYLES=${#STYLE_ARR[@]}

for STYLE_RAW in "${STYLE_ARR[@]}"; do
    STYLE="$(echo "$STYLE_RAW" | xargs)"
    if [ -z "$STYLE" ]; then
        continue
    fi

    if [ "$NUM_STYLES" -gt 1 ]; then
        STYLE_OUTPUT_FILE="${OUTPUT_FILE_TEMPLATE//\{style\}/$STYLE}"
        STYLE_ERROR_FILE="${ERROR_FILE_TEMPLATE//\{style\}/$STYLE}"
        STYLE_CHECKPOINT_FILE="${CHECKPOINT_FILE_TEMPLATE//\{style\}/$STYLE}"
    else
        STYLE_OUTPUT_FILE="$OUTPUT_FILE"
        STYLE_ERROR_FILE="$ERROR_FILE"
        STYLE_CHECKPOINT_FILE="$CHECKPOINT_FILE"
    fi

    echo "------------------------------------------"
    echo "Running prompt style: $STYLE"
    echo "Output: $STYLE_OUTPUT_FILE"
    echo "Error log: $STYLE_ERROR_FILE"
    echo "Checkpoint: $STYLE_CHECKPOINT_FILE"
    echo "------------------------------------------"

    python run_baselines.py \
        --api_base "$VLLM_API_BASE" \
        --api_key "dummy" \
        --model "$MODEL_NAME" \
        --input_path "$INPUT_PATH" \
        --output_path "$STYLE_OUTPUT_FILE" \
        --error_path "$STYLE_ERROR_FILE" \
        --checkpoint_path "$STYLE_CHECKPOINT_FILE" \
        --log_dir "$RUN_LOG_DIR" \
        --schema_file "${PROJECT_DIR}/data/schema.json" \
        --db_path "${PROJECT_DIR}/data/database.db" \
        --question_keys "natural_question" \
        --prompt_style "$STYLE" \
        --zs_prompt_file "$ZS_PROMPT_FILE" \
        --few_shot_file "$FEW_SHOT_FILE" \
        --batch_size "$BATCH_SIZE" \
        --batch_concurrency "$BATCH_CONCURRENCY" \
        --temperature 0.0 \
        --max_tokens 512 \
        --logprob_mode "$LOGPROB_MODE"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "Baseline inference failed for style=$STYLE with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
done

echo ""
echo "=========================================="
echo "Gemma-27B SQL baseline job completed."
if [ "$NUM_STYLES" -gt 1 ]; then
    echo "Results template: $OUTPUT_FILE_TEMPLATE"
else
    echo "Results: $OUTPUT_FILE"
fi
echo "=========================================="
