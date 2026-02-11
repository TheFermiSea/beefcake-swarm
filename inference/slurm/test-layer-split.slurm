#!/bin/bash
#SBATCH --job-name=test-layer
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=2
#SBATCH --nodelist=vasp-01,vasp-03
#SBATCH --gres=gpu:v100s:1
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/ai/logs/test-layer-%j.log
#SBATCH --error=/scratch/ai/logs/test-layer-%j.err

MODEL="/scratch/ai/models/or1-behemoth-q4_k_m.gguf"
CONTAINER="/cluster/shared/containers/llama-server.sif"
RPC_PORT=50052
SERVER_PORT=8082

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
RPC_NODE="${NODES[1]}"

echo "=== Layer Split Final Test ==="
echo "Testing with 64K context, layer split mode"
echo "RPC node: $RPC_NODE"

# Start RPC worker
srun --nodes=1 --ntasks=1 --nodelist="$RPC_NODE" --exclusive --gres=gpu:v100s:1 \
    apptainer exec --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    llama-rpc-server --host 0.0.0.0 --port $RPC_PORT &
sleep 20

# Start server with layer split
apptainer exec --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    llama-server \
        --model "$MODEL" \
        --rpc "${RPC_NODE}:${RPC_PORT}" \
        --split-mode layer \
        --host 0.0.0.0 \
        --port $SERVER_PORT \
        --ctx-size 65536 \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --n-gpu-layers 80 \
        --parallel 1 &
SERVER_PID=$!

# Wait for model to load (check health every 30s for up to 5 min)
echo "Waiting for model to load..."
for i in {1..10}; do
    sleep 30
    echo "Check $i ($(date))..."
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$SERVER_PORT/health 2>/dev/null)
    if [ "$HEALTH" = "200" ]; then
        echo "Server ready!"
        break
    fi
    echo "Status: $HEALTH (still loading)"
done

# Test inference
echo ""
echo "=== Testing Inference ==="
curl -s http://localhost:$SERVER_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello! Respond with exactly 3 words."}],"max_tokens":20}'

echo ""
echo ""
echo "=== Health Check ==="
curl -s http://localhost:$SERVER_PORT/health

echo ""
echo "=== TEST COMPLETE ==="
kill $SERVER_PID 2>/dev/null
