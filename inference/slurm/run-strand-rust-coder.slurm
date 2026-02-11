#!/bin/bash
#SBATCH --job-name=strand-rust-coder
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=vasp-02
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/strand-rust-coder-%j.log
#SBATCH --error=/scratch/ai/logs/strand-rust-coder-%j.err

###############################################################################
# Strand-Rust-Coder Inference Server (llama.cpp)
#
# Model: strand-rust-coder-14b-q8_0.gguf (~15GB)
# Hardware: V100S 32GB
#
# This job is preemptible - VASP HPC jobs have priority.
###############################################################################

set -euo pipefail

# Configuration
MODEL_PATH="${MODEL_PATH:-/scratch/ai/models/strand-rust-coder-14b-q8_0.gguf}"
CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-16}"

# Endpoint registration
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-strand-rust-coder.json"

# Logs
LOG_DIR="/scratch/ai/logs"
mkdir -p "${LOG_DIR}"

# Startup banner
echo "==========================================="
echo "Strand-Rust-Coder Inference Server"
echo "==========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "Model:        ${MODEL_PATH}"
echo "Port:         ${PORT}"
echo "Context:      ${CTX_SIZE} tokens"
echo "Threads:      ${THREADS}"
echo "==========================================="

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

    echo "[$(date)] Waiting for server to complete..."
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

# Validate container exists
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found at $CONTAINER"
    exit 1
fi

# Show model size
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "[$(date)] Model size: ${MODEL_SIZE}"

# Register endpoint
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "Strand-Rust-Coder",
    "tier": "rust",
    "node": "${SLURM_NODELIST}",
    "host": "$(hostname -f)",
    "port": ${PORT},
    "context_size": ${CTX_SIZE},
    "endpoint": "http://$(hostname -f):${PORT}/v1/chat/completions",
    "models_endpoint": "http://$(hostname -f):${PORT}/v1/models",
    "started_at": "$(date -Iseconds)",
    "notes": "14B Rust specialist model"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

echo "[$(date)] Starting llama-server..."

apptainer run --nv \
    --bind "$(dirname ${MODEL_PATH})":"$(dirname ${MODEL_PATH})":ro \
    --bind "${LOG_DIR}":"${LOG_DIR}":rw \
    "$CONTAINER" \
    --model "$MODEL_PATH" \
    --alias "Strand-Rust-Coder" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 \
    --threads "$THREADS" \
    --flash-attn auto \
    --temp 0.7 \
    --top-p 0.9 \
    --min-p 0.01 \
    --top-k 40 \
    --cont-batching \
    --metrics \
    --jinja &
SERVER_PID=$!

echo "[$(date)] Server started (PID $SERVER_PID)"
echo "[$(date)] Waiting for server to be ready..."

for i in {1..120}; do
    if curl -sf http://localhost:${PORT}/health &>/dev/null; then
        echo "[$(date)] Server is ready!"
        echo "[$(date)] API: http://$(hostname -f):${PORT}/v1/chat/completions"
        break
    fi
    sleep 5
    if [[ $i -eq 120 ]]; then
        echo "[$(date)] WARNING: Server not responding to health checks after 10 minutes"
        echo "[$(date)] Check logs for errors. Continuing anyway..."
    fi
done

wait
