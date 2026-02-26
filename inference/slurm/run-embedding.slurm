#!/bin/bash
#SBATCH --job-name=llama-embed
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=vasp-02
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-embed-%j.log
#SBATCH --error=/scratch/ai/logs/llama-embed-%j.err

###############################################################################
# CPU-Only Embedding Server - Pinned to vasp-02
#
# Purpose: Serve nomic-embed-code.Q8_0 for CocoIndex RAG embeddings.
# Runs CPU-only so it can coexist with the Qwen3.5 RPC worker (which uses
# vasp-02's GPU). The nomic model is ~150MB Q8_0 — trivial on CPU.
#
# Endpoint: http://vasp-02:8082/v1/embeddings
# Consumer: indexing/index_flow_v2.py (CocoIndex semantic search)
#
# No GPU requested — this job does not conflict with any GPU workload.
###############################################################################

set -euo pipefail

# Configuration
MODEL="${MODEL:-/scratch/ai/models/nomic-embed-code.Q8_0.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
PORT="${PORT:-8082}"
THREADS="${THREADS:-4}"

# Endpoint registration (for service discovery)
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-embed.json"

# Ensure log directory exists
mkdir -p /scratch/ai/logs

echo "=========================================="
echo "llama-server (embedding, CPU-only) starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "Threads: ${THREADS}"
echo "=========================================="

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
        echo "[$(date)] Signaling server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi

    echo "[$(date)] Waiting for server to complete gracefully..."
    wait

    rm -f "$ENDPOINT_FILE" 2>/dev/null || true
    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup EXIT

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Download with:"
    echo "  huggingface-cli download nomic-ai/nomic-embed-code-GGUF nomic-embed-code.Q8_0.gguf --local-dir /scratch/ai/models/"
    exit 1
fi

# Check binary exists
if [[ ! -x "$LLAMA_BIN/llama-server" ]]; then
    echo "ERROR: llama-server binary not found at $LLAMA_BIN/llama-server"
    exit 1
fi

# Register endpoint
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "mode": "embedding",
    "model": "nomic-embed-code.Q8_0",
    "tier": "embedding",
    "node": "${SLURM_NODELIST}",
    "host": "$(hostname -f)",
    "port": ${PORT},
    "endpoint": "http://$(hostname -f):${PORT}/v1/embeddings",
    "started_at": "$(date -Iseconds)"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

echo "[$(date)] Starting llama-server (embedding, CPU-only)..."

# Run embedding server — CPU-only, no GPU layers
"$LLAMA_BIN/llama-server" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --n-gpu-layers 0 \
    --embedding \
    --threads "$THREADS" \
    --ctx-size 8192 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cont-batching \
    --metrics &
SERVER_PID=$!
echo "[$(date)] Server started (PID $SERVER_PID)"

# Wait for server
wait
