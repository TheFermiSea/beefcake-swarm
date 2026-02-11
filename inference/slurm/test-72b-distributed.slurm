#!/bin/bash
#SBATCH --job-name=llama-72b-test
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ai/logs/llama-72b-test-%j.log
#SBATCH --error=/scratch/ai/logs/llama-72b-test-%j.err

###############################################################################
# Test: Distributed OR1-Behemoth (45GB Q4_K_M across 2x V100S = 64GB VRAM)
###############################################################################

set -euo pipefail

# Configuration - using Q4_K_M model
MODEL="/scratch/ai/models/or1-behemoth-q4_k_m.gguf"
CONTAINER="/cluster/shared/containers/llama-server.sif"
HEAD_PORT=8080
RPC_PORT=50052
CTX_SIZE=4096
THREADS=4

# Ensure log directory exists
mkdir -p /scratch/ai/logs

echo "=========================================="
echo "llama-72b distributed test starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "Model: ${MODEL}"
echo "Running on: $(hostname)"
echo "=========================================="

# Parse node list
mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE="${NODES[0]}"
RPC_NODE="${NODES[1]}"

echo "Head node: $HEAD_NODE"
echo "RPC worker: $RPC_NODE"

# Graceful shutdown
RPC_PID=""
HEAD_PID=""
cleanup() {
    echo "[$(date)] Shutting down..."
    [[ -n "$HEAD_PID" ]] && kill -TERM "$HEAD_PID" 2>/dev/null || true
    [[ -n "$RPC_PID" ]] && kill -TERM "$RPC_PID" 2>/dev/null || true
    wait
    echo "[$(date)] Shutdown complete"
}
trap cleanup SIGTERM SIGINT EXIT

# Verify model exists locally (we're running on head node)
echo "[$(date)] Verifying model availability locally..."
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found locally: $MODEL"
    ls -la /scratch/ai/models/ 2>/dev/null || echo "No /scratch/ai/models directory"
    exit 1
fi
echo "  Local model: OK ($(ls -lh "$MODEL" | awk '{print $5}'))"

# Prefetch model to page cache on local node
echo "[$(date)] Prefetching model to page cache..."
vmtouch -vt "$MODEL" 2>/dev/null || cat "$MODEL" > /dev/null

# Start RPC worker on second node using srun
echo "[$(date)] Starting RPC worker on $RPC_NODE:$RPC_PORT..."
srun --nodes=1 \
     --ntasks=1 \
     --nodelist="$RPC_NODE" \
     --exclusive \
     --gres=gpu:v100s:1 \
     --output="/scratch/ai/logs/rpc-${RPC_NODE}-${SLURM_JOB_ID}.log" \
     --error="/scratch/ai/logs/rpc-${RPC_NODE}-${SLURM_JOB_ID}.err" \
    apptainer exec --nv \
        --bind "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro \
        "$CONTAINER" \
        llama-rpc-server --host 0.0.0.0 --port "$RPC_PORT" &
RPC_PID=$!
echo "  RPC worker PID: $RPC_PID"

# Wait for RPC worker to be ready (use /dev/tcp from head node)
echo "[$(date)] Waiting for RPC worker on $RPC_NODE:$RPC_PORT..."
timeout_count=60
while ! timeout 2 bash -c "echo >/dev/tcp/$RPC_NODE/$RPC_PORT" 2>/dev/null; do
    sleep 1
    ((timeout_count--)) || true
    if [[ $timeout_count -le 0 ]]; then
        echo "ERROR: RPC worker failed to start after 60s"
        cat /scratch/ai/logs/rpc-${RPC_NODE}-${SLURM_JOB_ID}.log 2>/dev/null || true
        cat /scratch/ai/logs/rpc-${RPC_NODE}-${SLURM_JOB_ID}.err 2>/dev/null || true
        exit 1
    fi
    echo -n "."
done
echo ""
echo "  RPC worker ready!"

# Start head node server
echo "[$(date)] Starting head node llama-server on $HEAD_NODE:$HEAD_PORT..."
echo "  RPC connection: $RPC_NODE:$RPC_PORT"

apptainer run --nv \
    --bind "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro \
    --bind /scratch/ai/logs:/scratch/ai/logs:rw \
    "$CONTAINER" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$HEAD_PORT" \
    --rpc "${RPC_NODE}:${RPC_PORT}" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 \
    --threads "$THREADS" \
    --flash-attn on \
    --cont-batching &
HEAD_PID=$!

echo "[$(date)] Head node started (PID $HEAD_PID)"
echo "[$(date)] Waiting for model load (this may take a minute for 45GB model)..."

# Wait for server to be ready
timeout_count=180
while ! curl -sf "http://localhost:${HEAD_PORT}/health" >/dev/null 2>&1; do
    sleep 2
    ((timeout_count-=2)) || true
    if [[ $timeout_count -le 0 ]]; then
        echo "ERROR: Server failed to become ready after 3 minutes"
        exit 1
    fi
    echo -n "."
done
echo ""
echo "[$(date)] Server is READY!"
echo ""
echo "Endpoint: http://${HEAD_NODE}:${HEAD_PORT}/v1/chat/completions"
echo ""

# Run quick test
echo "[$(date)] Running inference test..."
response=$(curl -sf "http://localhost:${HEAD_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "or1-behemoth",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "max_tokens": 32,
        "temperature": 0.1
    }')

echo "Response: $response" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$response"

# Get usage stats
echo ""
echo "[$(date)] Performance metrics:"
curl -sf "http://localhost:${HEAD_PORT}/health" | jq . 2>/dev/null || true

echo ""
echo "[$(date)] Test complete! Server running. Press Ctrl+C or scancel to stop."
wait
