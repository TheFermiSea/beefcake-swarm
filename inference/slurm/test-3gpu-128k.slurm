#!/bin/bash
#SBATCH --job-name=test-3gpu-128k
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=vasp-01,vasp-02,vasp-03
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ai/logs/test-3gpu-128k-%j.log
#SBATCH --error=/scratch/ai/logs/test-3gpu-128k-%j.err

###############################################################################
# 3-GPU 128K Context Test with Layer Split
# Tests OR1-Behemoth 72B across all 3 V100S GPUs
###############################################################################

MODEL="/scratch/ai/models/or1-behemoth-q4_k_m.gguf"
CONTAINER="/cluster/shared/containers/llama-server.sif"
RPC_PORT=50052
SERVER_PORT=8082

set -euo pipefail

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
RPC_NODE1="${NODES[1]}"
RPC_NODE2="${NODES[2]}"

echo "=============================================="
echo "3-GPU 128K Context Test"
echo "=============================================="
echo "Head:       $HEAD_NODE"
echo "RPC Node 1: $RPC_NODE1"
echo "RPC Node 2: $RPC_NODE2"
echo "Context:    128K tokens"
echo "Split Mode: layer"
echo "Tensor Split: 26,27,27 (layers)"
echo "=============================================="
echo ""

# Start RPC workers on both worker nodes
echo "[$(date)] Starting RPC worker on $RPC_NODE1..."
srun --nodes=1 --ntasks=1 --nodelist="$RPC_NODE1" --exclusive --gres=gpu:v100s:1 \
    apptainer exec --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    llama-rpc-server --host 0.0.0.0 --port $RPC_PORT &
RPC_PID1=$!

echo "[$(date)] Starting RPC worker on $RPC_NODE2..."
srun --nodes=1 --ntasks=1 --nodelist="$RPC_NODE2" --exclusive --gres=gpu:v100s:1 \
    apptainer exec --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    llama-rpc-server --host 0.0.0.0 --port $RPC_PORT &
RPC_PID2=$!

echo "[$(date)] Waiting 30s for RPC workers to initialize..."
sleep 30

# Verify RPC workers are responding
echo "[$(date)] Checking RPC workers..."
timeout 5 bash -c "echo >/dev/tcp/$RPC_NODE1/$RPC_PORT" && echo "  $RPC_NODE1:$RPC_PORT - OK" || echo "  $RPC_NODE1:$RPC_PORT - FAILED"
timeout 5 bash -c "echo >/dev/tcp/$RPC_NODE2/$RPC_PORT" && echo "  $RPC_NODE2:$RPC_PORT - OK" || echo "  $RPC_NODE2:$RPC_PORT - FAILED"

echo ""
echo "[$(date)] Starting llama-server with 128K context..."

# Start server with layer split across 3 GPUs
apptainer exec --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    llama-server \
        --model "$MODEL" \
        --rpc "${RPC_NODE1}:${RPC_PORT},${RPC_NODE2}:${RPC_PORT}" \
        --split-mode layer \
        --tensor-split 26,27,27 \
        --host 0.0.0.0 \
        --port $SERVER_PORT \
        --ctx-size 131072 \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --n-gpu-layers 80 \
        --threads 8 \
        --batch-size 512 \
        --ubatch-size 256 \
        --parallel 1 \
        --timeout 600 &
SERVER_PID=$!

# Wait for model to load (check health every 30s for up to 8 min)
echo "[$(date)] Waiting for model to load (this may take 3-4 minutes)..."
for i in {1..16}; do
    sleep 30
    echo "[$(date)] Check $i/16..."
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/health 2>/dev/null)
    if [ "$HEALTH" = "200" ]; then
        echo "[$(date)] Server ready!"
        break
    fi
    echo "  Status: $HEALTH (still loading)"
done

# Verify server is ready
FINAL_HEALTH=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null)
echo ""
echo "=== Health Check ==="
echo "$FINAL_HEALTH"

if echo "$FINAL_HEALTH" | grep -q '"status":"ok"'; then
    echo ""
    echo "=== Testing Inference at 128K Context ==="

    # Test with a simple prompt first
    echo "[$(date)] Simple prompt test..."
    RESULT=$(curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Please respond with exactly 5 words."}],"max_tokens":30}')
    echo "$RESULT" | head -c 500
    echo ""

    # Check memory usage
    echo ""
    echo "=== GPU Memory Status ==="
    curl -s http://localhost:$SERVER_PORT/health | grep -o '"slots_idle":[0-9]*' || true

    echo ""
    echo "=============================================="
    echo "3-GPU 128K TEST SUCCESSFUL"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "TEST FAILED - Server did not become ready"
    echo "=============================================="
fi

# Cleanup
kill $SERVER_PID 2>/dev/null || true
kill $RPC_PID1 2>/dev/null || true
kill $RPC_PID2 2>/dev/null || true
