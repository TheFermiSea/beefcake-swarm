#!/bin/bash
#SBATCH --job-name=llama-122b-rpc
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=vasp-01,vasp-02
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-122b-%j.log
#SBATCH --error=/scratch/ai/logs/llama-122b-%j.err

###############################################################################
# Integrator Tier (Qwen3.5-122B-A10B MoE)
#
# Model: Qwen3.5-122B-A10B (~72GB Q4_K_M)
# Context: 128K tokens
# Hardware: 2x V100S 32GB (vasp-01 head + vasp-02 worker)
#
# Architecture:
# - Dual-Node RPC: Pools 64GB VRAM.
# - Layer Split: Optimized for V100S memory capacity.
# - MoE Sparsity: Only 10B parameters active per token.
# - Prompt Caching: Critical for multi-turn integration tasks.
###############################################################################

set -euo pipefail

# Configuration
MODEL_DIR="/scratch/ai/models"
MODEL_PATH="${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
PORT="${PORT:-8081}"
RPC_PORT="${RPC_PORT:-50052}"
CTX_SIZE="${CTX_SIZE:-131072}"
THREADS="${THREADS:-16}"

# Parse node list
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

# Endpoint registration
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-122b.json"

# Ensure log directory exists
mkdir -p /scratch/ai/logs
SHARED_LOG_DIR="/cluster/shared/ai/logs/${SLURM_JOB_ID}"
mkdir -p "$SHARED_LOG_DIR"

# Graceful shutdown handler
RPC_PID=""
HEAD_PID=""
SHUTTING_DOWN=false

cleanup() {
    if $SHUTTING_DOWN; then
        return
    fi
    SHUTTING_DOWN=true
    echo "[$(date)] SIGTERM received - initiating graceful shutdown"
    if [[ -n "$HEAD_PID" ]]; then
        kill -TERM "$HEAD_PID" 2>/dev/null || true
    fi
    if [[ -n "$RPC_PID" ]]; then
        kill -TERM "$RPC_PID" 2>/dev/null || true
    fi
    wait
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true
    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup SIGTERM SIGINT

# Prefetch model on both nodes
echo "[$(date)] Prefetching model shards..."
for node in "${NODES[@]}"; do
    ssh -o BatchMode=yes "$node" "numactl --interleave=all sh -c 'vmtouch -vt ${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-0000* > /dev/null 2>&1 || cat ${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M-0000* > /dev/null'" &
done
wait

# Start RPC worker on vasp-02
echo "[$(date)] Starting RPC worker on ${WORKER_NODE}..."
srun --nodes=1 --ntasks=1 --nodelist="${WORKER_NODE}" --exclusive --gres=gpu:v100s:1 \
    bash -c "LD_LIBRARY_PATH='$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB' numactl --interleave=all '$LLAMA_BIN/rpc-server' --host 0.0.0.0 --port '$RPC_PORT'" > "${SHARED_LOG_DIR}/rpc-worker.log" 2>&1 &
RPC_PID=$!

# Wait for RPC port
echo "[$(date)] Waiting for RPC worker..."
for i in {1..60}; do
    if timeout 2 bash -c "echo >/dev/tcp/${WORKER_NODE}/${RPC_PORT}" 2>/dev/null; then
        echo "[$(date)] RPC worker ready!"
        break
    fi
    sleep 2
done

# Fail hard if RPC worker never came up
if ! timeout 2 bash -c "echo >/dev/tcp/${WORKER_NODE}/${RPC_PORT}" 2>/dev/null; then
    echo "[$(date)] ERROR: RPC worker on ${WORKER_NODE}:${RPC_PORT} failed to start after 120s"
    cleanup
    exit 1
fi

# Register endpoint
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "Qwen3.5-122B-A10B",
    "tier": "integrator",
    "head_node": "${HEAD_NODE}",
    "worker_node": "${WORKER_NODE}",
    "host": "${HEAD_NODE}",
    "port": ${PORT},
    "context_size": ${CTX_SIZE},
    "endpoint": "http://${HEAD_NODE}:${PORT}/v1/chat/completions",
    "started_at": "$(date -Iseconds)",
    "notes": "122B MoE Integrator, 128K Distributed RPC Context"
}
EOF

echo "[$(date)] Starting Integrator Head on ${HEAD_NODE}..."

# Launch Head Node
export LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_FORCE_MMQ=1
# Prevent V100 MoE routing crash (same bug as Qwen3.5-397B — the 122B is also MoE)
export GGML_CUDA_DISABLE_FUSION=1

numactl --interleave=all "$LLAMA_BIN/llama-server" \
    --model "$MODEL_PATH" \
    --alias "Qwen3.5-122B-A10B" \
    --rpc "${WORKER_NODE}:${RPC_PORT}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --split-mode layer \
    --n-gpu-layers 99 \
    --threads "$THREADS" \
    --batch-size 512 \
    --ubatch-size 512 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --prompt-cache "${MODEL_DIR}/122b-prompt-cache.bin" \
    --prompt-cache-all \
    --mlock \
    --cont-batching \
    --metrics \
    --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' &
HEAD_PID=$!

echo "[$(date)] Integrator started (PID $HEAD_PID)"
wait
