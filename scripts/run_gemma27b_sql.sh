#!/bin/bash
#SBATCH --job-name=gemma27b-sql
#SBATCH --account=3dllms
#SBATCH --partition=inv-ssheshap
#SBATCH --gres=gpu:2
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=/gscratch/tkishore/MAYO_AIM2/logs/gemma27b_with_conf_%j.out
#SBATCH --error=/gscratch/tkishore/MAYO_AIM2/logs/gemma27b_with_conf_%j.err

# ==========================================
# CONFIG
# ==========================================

MODEL_NAME="google/gemma-3-27b-it"
TP_SIZE=2
GPU_MEM=0.85

VLLM_PORT=8000
VLLM_HOST="127.0.0.1"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

PROJECT_DIR="/gscratch/tkishore/MAYO_AIM2/"
LOG_DIR="${PROJECT_DIR}/logs/Gemma/27B/sql_with_conf"
mkdir -p "$LOG_DIR"

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
echo "🚀 Starting Gemma-3-27B SQL Job"
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
        echo "⏳ Waiting for vLLM server... ($attempt/$max_attempts)"
        sleep 5
    done
    echo "❌ vLLM server failed to start"
    return 1
}

cleanup() {
    echo ""
    echo "🧹 Cleaning up vLLM server..."
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
echo "🔧 Launching vLLM server..."

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
    --structured-outputs-config.backend xgrammar \
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
# RUN INFERENCE
# ==========================================

OUTPUT_FILE="${LOG_DIR}/generated_sql.jsonl"
ERROR_FILE="${LOG_DIR}/error_log.jsonl"

echo ""
echo "📊 Running SQL generation..."
echo "Output: $OUTPUT_FILE"
echo ""

python vllm_api_inference.py \
    --api_base "$VLLM_API_BASE" \
    --api_key "dummy" \
    --model_name "$MODEL_NAME" \
    --input_file data/dataset.jsonl \
    --output_file "$OUTPUT_FILE" \
    --error_file "$ERROR_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "✅ Gemma-27B SQL job completed!"
echo "Results: $OUTPUT_FILE"
echo "=========================================="
