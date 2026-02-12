#!/bin/bash
#SBATCH --job-name=llama-14b
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=vasp-02
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-14b-%j.log
#SBATCH --error=/scratch/ai/logs/llama-14b-%j.err

###############################################################################
# Multi-Model Inference Server (Router Mode) - Pinned to vasp-02
#
# Purpose: Fast inference (strand-14B) + general coding (Qwen3-Coder-Next)
#
# Models:
#   strand-14B:          "Mechanic" — fast Rust fixes, ~53 tok/s, <1s TTFT
#   Qwen3-Coder-Next:   "Implementer" — multi-file coding, 256K ctx, ~5-15 tok/s
#
# Topology: vasp-02 serves fast+coder tiers. vasp-01+03 handle reasoning (72B).
#
# Router mode: Serves multiple models from --models-dir, loads on-demand
# based on "model" field in API requests. Uses LRU eviction when at capacity.
#
# MoE offloading: -ot flag pushes Qwen3-Coder-Next's MoE expert FFN layers
# to CPU. Harmless for dense models like strand-14B (no matching tensors).
# VRAM budget: ~15GB strand + ~8GB Qwen3 attention + ~1-3GB KV (q4_0) = ~24-26GB of 32GB.
# CPU budget: ~40GB for MoE weights + OS headroom.
# Context: 64K tokens shared across all models. KV cache uses q4_0 quantization.
#
# Thread allocation (32 total):
#   MoE expert computation: 16 threads
#   CocoIndex RAG:          4 threads
#   cargo build/test:       8 threads
#   OS headroom:            4 threads
#
# This job is preemptible - VASP jobs have priority. The server will be
# gracefully terminated (30s warning) when a VASP job needs the GPU.
###############################################################################

set -euo pipefail

# Configuration
MODELS_DIR="${MODELS_DIR:-/scratch/ai/models}"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-65536}"
THREADS="${THREADS:-32}"
MODELS_MAX="${MODELS_MAX:-2}"  # Max concurrent models in VRAM

# Endpoint registration (for service discovery)
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-14b.json"

# Ensure log directory exists
mkdir -p /scratch/ai/logs

# Log startup
echo "=========================================="
echo "llama-server (router mode) starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Models Dir: ${MODELS_DIR}"
echo "Max Models: ${MODELS_MAX}"
echo "Port: ${PORT}"
echo "Context Size: ${CTX_SIZE}"
echo "=========================================="

# Graceful shutdown handler for SLURM preemption
# Track server PID for clean shutdown
SERVER_PID=""
SHUTTING_DOWN=false

cleanup() {
    # Prevent re-entry
    if $SHUTTING_DOWN; then
        return
    fi
    SHUTTING_DOWN=true

    echo "[$(date)] SIGTERM received - initiating graceful shutdown"

    # Signal the server process
    if [[ -n "$SERVER_PID" ]]; then
        echo "[$(date)] Signaling server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi

    # Wait for graceful completion
    echo "[$(date)] Waiting for server to complete gracefully..."
    wait

    # Remove endpoint file
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true

    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup SIGTERM SIGINT

# Check models directory exists
if [[ ! -d "$MODELS_DIR" ]]; then
    echo "ERROR: Models directory not found at $MODELS_DIR"
    exit 1
fi

# List available models
echo "[$(date)] Available models in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (no .gguf files found)"

# Prefetch commonly used models to page cache
DEFAULT_MODEL="$MODELS_DIR/strand-rust-coder-14b-q8_0.gguf"
CODER_MODEL="$MODELS_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
for model in "$DEFAULT_MODEL" "$CODER_MODEL"; do
    if [[ -f "$model" ]]; then
        echo "[$(date)] Prefetching $(basename "$model") to page cache..."
        if command -v vmtouch &> /dev/null; then
            vmtouch -vt "$model" 2>/dev/null || true
        else
            cat "$model" > /dev/null 2>&1 || true
        fi
    fi
done

# Check binary exists
if [[ ! -x "$LLAMA_BIN/llama-server" ]]; then
    echo "ERROR: llama-server binary not found at $LLAMA_BIN/llama-server"
    exit 1
fi

# Register endpoint on shared filesystem (for service discovery)
# Using /cluster/shared ensures endpoint is visible from slurm-ctl
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "mode": "router",
    "models_dir": "${MODELS_DIR}",
    "models": ["strand-rust-coder-14b-q8_0", "Qwen3-Coder-Next"],
    "tier": "fast+coder",
    "node": "${SLURM_NODELIST}",
    "host": "$(hostname -f)",
    "port": ${PORT},
    "endpoint": "http://$(hostname -f):${PORT}/v1/chat/completions",
    "anthropic_endpoint": "http://$(hostname -f):${PORT}/v1/messages",
    "models_endpoint": "http://$(hostname -f):${PORT}/v1/models",
    "started_at": "$(date -Iseconds)"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

# Note: cleanup() handles endpoint file removal and is already trapped for SIGTERM SIGINT

echo "[$(date)] Starting llama-server..."

# Run inference server in router mode (native binary, no container)
# --models-dir: Serve multiple models from directory
# --models-autoload: Load models on-demand based on API requests
# -ot: MoE expert FFN layer offloading to CPU (Qwen3-Coder-Next).
#       Harmless for dense models like strand-14B (no matching tensors).
# --jinja: Jinja2 chat template support (required by Qwen3-Coder-Next)
LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$LLAMA_BIN/llama-server" \
    --models-dir "$MODELS_DIR" \
    --models-max "$MODELS_MAX" \
    --models-autoload \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --threads "$THREADS" \
    --flash-attn on \
    --jinja \
    --embedding \
    --cont-batching \
    --metrics \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 &
SERVER_PID=$!
echo "[$(date)] Server started (PID $SERVER_PID)"

# Wait for server - this keeps the job alive and enables graceful shutdown
wait
