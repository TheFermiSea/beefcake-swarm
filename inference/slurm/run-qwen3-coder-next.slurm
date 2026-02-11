#!/bin/bash
#SBATCH --job-name=qwen3-coder
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=vasp-01
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/qwen3-coder-%j.log
#SBATCH --error=/scratch/ai/logs/qwen3-coder-%j.err
 
 ###############################################################################
 # Qwen3-Coder-Next with MoE CPU Offloading (Review Tier)
 #
 # Time-shared with OR1-Behemoth (Tier 2). Only runs when Tier 2 Integrator is idle.
 #
 # Model: 80B total params, 3B activated (MoE with 512 experts, 10 active)
 # Architecture: Hybrid Gated DeltaNet + Attention + MoE
 # Context: 256K native (using 32K default for memory efficiency)
 #
 # Hardware Strategy:
 #   - GPU (32GB V100S): Attention layers, shared expert
 #   - CPU (256GB RAM): MoE expert layers via -ot offloading
 #   - 32 threads: MoE computation on CPU
 #
 # This job is preemptible - VASP HPC jobs have priority. The server will be
 # gracefully terminated (30s warning) when a VASP job needs the GPU.
 # Additionally preempted by the reasoning tier (72B) which has higher QoS priority.
 #
 # Performance:
 #   - Expected: 5-15 tok/s with MoE offloading
 #   - TTFT: 2-5s depending on prompt length
 #   - Context: Up to 32K tokens (can increase if RAM allows)
 #
 # Recommended sampling (per Qwen):
 #   temperature=1.0, top_p=0.95, top_k=40, min_p=0.01
 ###############################################################################
 
 set -euo pipefail
 
 # Configuration
 MODEL_PATH="${MODEL_PATH:-/scratch/ai/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
 CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
 PORT="${PORT:-8002}"
CTX_SIZE="${CTX_SIZE:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
 THREADS="${THREADS:-32}"
 
 # MoE offloading: Push all MoE expert layers to CPU
 # Pattern explanation: .ffn_.*_exps. matches all MoE expert feed-forward layers
 # This keeps attention layers on GPU for speed, experts on CPU for capacity
 MOE_OFFLOAD="${MOE_OFFLOAD:-.ffn_.*_exps.=CPU}"
 
 # Endpoint registration
 ENDPOINT_DIR="/cluster/shared/ai/endpoints"
 ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-qwen3-coder.json"
 
 # Logs
 LOG_DIR="/scratch/ai/logs"
 mkdir -p "${LOG_DIR}"
 
 # Startup banner
 echo "==========================================="
 echo "Qwen3-Coder-Next MoE Inference Server"
 echo "==========================================="
 echo "Job ID:       ${SLURM_JOB_ID}"
 echo "Node:         ${SLURM_NODELIST}"
 echo "Model:        ${MODEL_PATH}"
 echo "Port:         ${PORT}"
 echo "Context:      ${CTX_SIZE} tokens"
echo "Max output:   ${MAX_TOKENS} tokens"
 echo "Threads:      ${THREADS}"
 echo "MoE Offload:  ${MOE_OFFLOAD}"
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
     
     # Remove endpoint registration
     rm -f "$ENDPOINT_FILE" 2>/dev/null || true
     
     echo "[$(date)] Graceful shutdown complete"
 }
 trap cleanup SIGTERM SIGINT
 
 # Validate model exists
 if [[ ! -f "$MODEL_PATH" ]]; then
     echo "ERROR: Model not found at $MODEL_PATH"
     echo ""
     echo "Download with:"
     echo "  huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \\"
     echo "    --local-dir /scratch/ai/models/Qwen3-Coder-Next-GGUF \\"
     echo "    --include '*UD-Q4_K_XL*'"
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
 
 # Prefetch model to page cache
 echo "[$(date)] Prefetching model to page cache..."
 if command -v vmtouch &> /dev/null; then
     vmtouch -vt "$MODEL_PATH" 2>/dev/null || true
 else
     cat "$MODEL_PATH" > /dev/null 2>&1 || true
 fi
 
 # Show system state
 echo "[$(date)] System memory before load:"
 free -h
 echo ""
 echo "[$(date)] GPU state before load:"
 nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv
 echo ""
 
 # Register endpoint
 mkdir -p "${ENDPOINT_DIR}"
 cat > "$ENDPOINT_FILE" <<EOF
 {
     "job_id": "${SLURM_JOB_ID}",
     "model": "Qwen3-Coder-Next",
     "mode": "moe-offload",
     "tier": "coder",
     "node": "${SLURM_NODELIST}",
     "host": "$(hostname -f)",
     "port": ${PORT},
     "context_size": ${CTX_SIZE},
     "endpoint": "http://$(hostname -f):${PORT}/v1/chat/completions",
     "models_endpoint": "http://$(hostname -f):${PORT}/v1/models",
     "started_at": "$(date -Iseconds)",
     "notes": "80B MoE model with 3B active params, MoE layers on CPU"
 }
EOF
 echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"
 
 echo "[$(date)] Starting llama-server with MoE CPU offloading..."
 echo "[$(date)] This may take 2-5 minutes for initial model loading..."
 
 # Run inference server
 # Key flags:
 #   -ot: Offload tensor layers matching pattern to CPU
 #   --threads: CPU threads for offloaded computation
 #   --flash-attn: Efficient attention (if supported by container)
 #   --cont-batching: Continuous batching for throughput
 #   --jinja: Jinja2 template support for chat templates
 apptainer run --nv \
     --bind "$(dirname ${MODEL_PATH})":"$(dirname ${MODEL_PATH})":ro \
     --bind "${LOG_DIR}":"${LOG_DIR}":rw \
     "$CONTAINER" \
     --model "$MODEL_PATH" \
     --alias "Qwen3-Coder-Next" \
     -ot "$MOE_OFFLOAD" \
     --host 0.0.0.0 \
     --port "$PORT" \
     --ctx-size "$CTX_SIZE" \
    --n-predict "$MAX_TOKENS" \
     --n-gpu-layers 99 \
     --threads "$THREADS" \
    --flash-attn auto \
     --temp 1.0 \
     --top-p 0.95 \
     --min-p 0.01 \
     --top-k 40 \
     --cont-batching \
     --metrics \
     --jinja &
 SERVER_PID=$!
 
 echo "[$(date)] Server started (PID $SERVER_PID)"
 echo "[$(date)] Waiting for server to be ready..."
 
 # Wait for server to be healthy
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
 
 # Show final memory state
 echo ""
 echo "[$(date)] System memory after load:"
 free -h
 echo ""
 echo "[$(date)] GPU state after load:"
 nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv
 
 # Keep job alive
 wait
