#!/bin/bash
#SBATCH --job-name=llama-122b
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-122b-%j.log
#SBATCH --error=/scratch/ai/logs/llama-122b-%j.err

###############################################################################
# Integrator Tier (Qwen3.5-122B-A10B MoE) — Single-Node Expert Offload
#
# Model: Qwen3.5-122B-A10B (~72GB Q4_K_M, MoE: 64 experts, 8 active)
# Context: 65K tokens (fits in 32GB VRAM with experts on CPU)
# Hardware: V100S 32GB + 200GB system RAM
#
# Architecture:
# - Expert Offload: -ot .ffn_.*_exps.=CPU puts ~60GB experts in RAM
# - Dense on GPU: ~12GB attention/routing/norm layers fully on V100S
# - MoE Sparsity: Only 10B parameters active per token
# - Expected: 8-12 tok/s generation, ~40 tok/s prompt processing
#
# Why single-node over RPC:
# - Expert offload means dense footprint is only ~12GB (fits 1 GPU)
# - No RPC network latency for thousands of tensor transfers per token
# - Simpler, more stable, and actually faster than dual-node RPC
# - Submit this script on BOTH vasp-01 and vasp-02 for 4 concurrent slots
###############################################################################

set -euo pipefail

# Configuration
MODEL_DIR="/scratch/ai/models"
MODEL_PATH="${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
MATH_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib"
PORT="${PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-65536}"
THREADS="${THREADS:-32}"

# Endpoint registration
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-122b.json"

# Ensure log directory exists
mkdir -p /scratch/ai/logs

# Graceful shutdown handler
SERVER_PID=""
SHUTTING_DOWN=false

cleanup() {
    if $SHUTTING_DOWN; then
        return
    fi
    SHUTTING_DOWN=true
    echo "[$(date)] SIGTERM received - initiating graceful shutdown"
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    wait
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true
    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup SIGTERM SIGINT

# Validate model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Prefetch model shards into page cache
echo "[$(date)] Prefetching model shards..."
numactl --interleave=all sh -c "vmtouch -vt ${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-0000* > /dev/null 2>&1 || cat ${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-0000* > /dev/null"

# Register endpoint
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "Qwen3.5-122B-A10B",
    "tier": "integrator",
    "node": "${SLURM_NODELIST}",
    "host": "$(hostname -f)",
    "port": ${PORT},
    "context_size": ${CTX_SIZE},
    "endpoint": "http://$(hostname -f):${PORT}/v1/chat/completions",
    "started_at": "$(date -Iseconds)",
    "notes": "122B MoE Integrator, 65K Context, Expert-Offload Single-Node"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

echo "[$(date)] Starting llama-server..."

# LD_LIBRARY_PATH for native binary
export LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB:$MATH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Prevent V100 MoE routing crash
export GGML_CUDA_DISABLE_FUSION=1

numactl --interleave=all "$LLAMA_BIN/llama-server" \
    --model "$MODEL_PATH" \
    --alias "Qwen3.5-122B-A10B" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --threads "$THREADS" \
    --batch-size 512 \
    --ubatch-size 512 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --parallel 2 \
    --mlock \
    --cont-batching \
    --metrics \
    --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' &
SERVER_PID=$!

echo "[$(date)] Server started (PID $SERVER_PID)"
wait
