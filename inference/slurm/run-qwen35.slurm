#!/bin/bash
#SBATCH --job-name=llama-qwen35
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-qwen35-%j.log
#SBATCH --error=/scratch/ai/logs/llama-qwen35-%j.err

###############################################################################
# Single-Node Qwen3.5-397B-A17B Inference Server (MoE)
#
# Parameterized script for running independent Qwen3.5 instances per node.
# Replaces the 3-node distributed (RPC) setup — single-node is 60% faster
# (8.4 vs 5.1 tok/s) because RPC overhead was the bottleneck.
#
# Model: Qwen3.5-397B-A17B (~214GB at Q4_K_XL, ~241GB at Q4_K_M)
# Throughput: ~8-10 tok/s gen, ~21-40 tok/s prompt (varies by backend/quant)
# Context: up to 256K tokens native (no YaRN needed)
#
# Supports two backends:
#   mainline: Standard llama.cpp — MoE experts on CPU via -ot flag
#   ik:       ik_llama.cpp fork — proper hybrid CPU/GPU MoE via --cpu-moe flag
#             Required for GPU inference until mainline fixes qwen35moe CUDA (#19683)
#
# Memory budget (single node, 256GB RAM):
#   Model weights: ~214-241GB (GPU layers + CPU MoE experts)
#   KV cache: allocated on demand (~8GB per slot at 128K q4_0, ~4GB at 65K)
#   OS + headroom: ~16GB
#
# CPU allocation (36 cores/node):
#   MoE expert computation dominates — all threads available
#   --threads for decode, --threads-batch for prompt/prefill
#
# NUMA topology (4 vNUMA nodes per VM, matching host SNC):
#   vNUMA 0-1: Physical Socket 0 (GPU-local), ~64GB each
#   vNUMA 2-3: Physical Socket 1, ~64GB each
#   Memory interleaved across all NUMA nodes (numactl --interleave=all)
#   for maximum aggregate bandwidth on CPU-bound MoE expert computation.
#
# Usage:
#   # Architect (vasp-01): 2 slots, 128K context, mainline backend
#   sbatch --nodelist=vasp-01 \
#     --export=ALL,PORT=8081,PARALLEL_SLOTS=2,CTX_SIZE=131072,ENDPOINT_SUFFIX=qwen35,TIER_NAME=manager-local \
#     run-qwen35.slurm
#
#   # Implementer (vasp-02): 4 slots, 65K context, ik_llama.cpp backend
#   sbatch --nodelist=vasp-02 \
#     --export=ALL,PORT=8080,PARALLEL_SLOTS=4,CTX_SIZE=65536,ENDPOINT_SUFFIX=qwen35-impl,TIER_NAME=implementer,LLAMA_BACKEND=ik \
#     run-qwen35.slurm
#
# This job is preemptible — VASP jobs have priority.
###############################################################################

set -euo pipefail

# Configuration (all parameterized via env vars with sensible defaults)
MODEL="${MODEL:-/scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf}"
# Backend selection: "mainline" (default) or "ik" (ik_llama.cpp fork with --cpu-moe).
# ik_llama.cpp properly handles hybrid CPU/GPU MoE inference, required until
# mainline fixes qwen35moe CUDA compute graph (issue #19683).
LLAMA_BACKEND="${LLAMA_BACKEND:-mainline}"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
IK_LLAMA_BIN="${IK_LLAMA_BIN:-/cluster/shared/llama-cpp/ik-bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
PORT="${PORT:-8081}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-2}"
CTX_SIZE="${CTX_SIZE:-131072}"           # Per-slot context (128K default)
N_PREDICT="${N_PREDICT:--1}"             # -1 = unlimited (reasoning models need full CoT)
THREADS="${THREADS:-28}"                 # Decode threads (fewer than max reduces memory bandwidth contention)
THREADS_BATCH="${THREADS_BATCH:-36}"     # Batch/prompt threads (use all cores for prefill)
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
ENDPOINT_SUFFIX="${ENDPOINT_SUFFIX:-qwen35}"
TIER_NAME="${TIER_NAME:-manager-local}"
# Enable/disable --jinja template processing. Set JINJA_MODE=off to disable.
# --jinja may cause degenerate output on some quantizations (UD-Q4_K_XL).
JINJA_MODE="${JINJA_MODE:-on}"
# GPU layer offload count. Set to 0 for CPU-only inference.
# llama.cpp qwen35moe CUDA compute graph is buggy (issue #19683) — GPU layers
# produce degenerate output. CPU-only (-ngl 0) is the known workaround.
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
# Chat-to-completion proxy port (proxy translates /v1/chat/completions → /v1/completions).
# Qwen3.5 at Q4_K_XL generates immediate EOS for any instruction-following prompt format
# through chat/completions, but works fine via /v1/completions with continuation-style prompts.
# Default: PORT + 100 (e.g., 8180 for PORT=8080, 8181 for PORT=8081).
PROXY_PORT="${PROXY_PORT:-$((PORT + 100))}"
PROXY_SCRIPT="${PROXY_SCRIPT:-/cluster/shared/scripts/llama-cpp/chat-proxy/proxy.py}"

# Ensure log directory exists
mkdir -p /scratch/ai/logs

# Log startup
echo "=========================================="
echo "llama-qwen35 single-node inference starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname -s)"
echo "Backend: ${LLAMA_BACKEND}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "Parallel Slots: ${PARALLEL_SLOTS}"
echo "Context Size: ${CTX_SIZE} (per slot)"
echo "Tier: ${TIER_NAME}"
echo "GPU Layers: ${N_GPU_LAYERS}"
echo "Proxy Port: ${PROXY_PORT}"
echo "Endpoint Suffix: ${ENDPOINT_SUFFIX}"
echo "Threads: ${THREADS}"
echo "Threads (batch): ${THREADS_BATCH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Ubatch size: ${UBATCH_SIZE}"
echo "=========================================="

###############################################################################
# Memory Validation
###############################################################################

MODEL_SIZE_GB=214
# KV cache size scales with context: ~8GB per slot at 128K q4_0, ~4GB at 65K
# Formula: ctx_size * 2 (K+V) * n_layers * d_head * n_heads * q4_0_bytes / 1GB
# Approximate: 8GB per 128K slot with q4_0 quantization
KV_PER_SLOT_GB=$(( (CTX_SIZE * 8 + 131071) / 131072 ))  # ~8GB per 128K, ~4GB per 65K
OS_HEADROOM_GB=16

TOTAL_KV_GB=$((KV_PER_SLOT_GB * PARALLEL_SLOTS))
TOTAL_MEMORY_GB=$((MODEL_SIZE_GB + TOTAL_KV_GB + OS_HEADROOM_GB))

echo "[$(date)] Memory estimate: ${MODEL_SIZE_GB}GB model + ${TOTAL_KV_GB}GB KV (${PARALLEL_SLOTS} slots @ ${KV_PER_SLOT_GB}GB each) + ${OS_HEADROOM_GB}GB OS = ${TOTAL_MEMORY_GB}GB total"

if [[ $TOTAL_MEMORY_GB -gt 256 ]]; then
    echo "[$(date)] WARNING: Memory estimate (${TOTAL_MEMORY_GB}GB) exceeds node capacity (256GB)"
    echo "[$(date)] KV cache is allocated on demand — this may still work if not all slots are filled simultaneously"
    echo "[$(date)] Reduce PARALLEL_SLOTS or CTX_SIZE if OOM occurs"
fi

if [[ $TOTAL_MEMORY_GB -gt 240 ]]; then
    echo "[$(date)] NOTE: Memory estimate (${TOTAL_MEMORY_GB}GB) exceeds SLURM allocation (240GB)"
    echo "[$(date)] KV cache is demand-allocated — proceeding (will only OOM if all slots saturated)"
fi

# Check model availability
echo "[$(date)] Checking model availability..."
if [[ ! -f "$MODEL" ]]; then
    echo "[$(date)] ERROR: Model not found at $MODEL"
    echo "Model is stored on local /scratch (not NFS), so it must be present on this node."
    echo ""
    echo "Download or copy the model:"
    echo "  # Download from HuggingFace (split GGUF, 6 shards):"
    echo "  python3.11 -c \"from huggingface_hub import snapshot_download;"
    echo "    snapshot_download('unsloth/Qwen3.5-397B-A17B-GGUF',"
    echo "      local_dir='/scratch/ai/models/Qwen3.5-397B-A17B-GGUF',"
    echo "      allow_patterns=['UD-Q4_K_XL/*'])\""
    echo ""
    echo "  # Or copy from another node:"
    echo "  rsync -avP root@vasp-01:/scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/ \\"
    echo "    /scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/"
    exit 1
fi
echo "  $(hostname -s): OK"

# Prefetch model to page cache
# NOTE: No --mlock — model is ~214GB, too large to lock. vmtouch prefetch is sufficient.
echo "[$(date)] Prefetching model to page cache..."
numactl --interleave=all sh -c "vmtouch -vt '$MODEL' 2>/dev/null || cat '$MODEL' > /dev/null"
echo "[$(date)] Prefetch complete"

# Check for port conflicts
if timeout 2 bash -c "echo >/dev/tcp/localhost/${PORT}" 2>/dev/null; then
    echo "[$(date)] WARNING: Port $PORT already in use on $(hostname -s) — may conflict"
fi

###############################################################################
# Endpoint Registration
###############################################################################

ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-${ENDPOINT_SUFFIX}.json"
mkdir -p "${ENDPOINT_DIR}"

NODE_NAME="$(hostname -s)"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "Qwen3.5-397B-A17B",
    "tier": "${TIER_NAME}",
    "host": "${NODE_NAME}",
    "port": ${PORT},
    "endpoint": "http://${NODE_NAME}:${PROXY_PORT}/v1/chat/completions",
    "backend_port": ${PORT},
    "proxy_port": ${PROXY_PORT},
    "parallel_slots": ${PARALLEL_SLOTS},
    "context_per_slot": ${CTX_SIZE},
    "started_at": "$(date -Iseconds)"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

###############################################################################
# Graceful Shutdown
###############################################################################

SERVER_PID=""
PROXY_PID=""
SHUTTING_DOWN=false

cleanup() {
    if $SHUTTING_DOWN; then
        return
    fi
    SHUTTING_DOWN=true

    echo "[$(date)] SIGTERM received — initiating graceful shutdown"

    if [[ -n "$PROXY_PID" ]]; then
        echo "[$(date)] Signaling chat proxy (PID $PROXY_PID)..."
        kill -TERM "$PROXY_PID" 2>/dev/null || true
    fi

    if [[ -n "$SERVER_PID" ]]; then
        echo "[$(date)] Signaling llama-server (PID $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi

    # Wait for processes to exit gracefully
    wait "$PROXY_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true

    # Remove endpoint file
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true

    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup SIGTERM SIGINT

###############################################################################
# Launch Server
###############################################################################

echo "[$(date)] Starting llama-server..."

SERVER_ARGS=(
    --model "$MODEL"
    --alias "Qwen3.5-397B-A17B"
    --host 0.0.0.0
    --port "$PORT"
    --ctx-size "$CTX_SIZE"
    --n-predict "$N_PREDICT"
    # GPU layers — set N_GPU_LAYERS=0 for CPU-only (workaround for qwen35moe CUDA bug #19683)
    --n-gpu-layers "$N_GPU_LAYERS"
    # Parallel slots for concurrent sessions
    --parallel "$PARALLEL_SLOTS"
    # Threading for MoE CPU computation
    --threads "$THREADS"
    --threads-batch "$THREADS_BATCH"
    # Continuous batching for multi-slot throughput
    --cont-batching
    # Jinja template processing — enables per-request chat_template_kwargs
    # (e.g., {"enable_thinking": false} for Qwen3.5 instruct/non-thinking mode).
    # Conditionally enabled via JINJA_MODE env var (default: on).
    # WARNING: --jinja may cause degenerate output with UD-Q4_K_XL quantization.
    # Quantized KV cache — q4_0 keeps large context manageable
    --cache-type-k q4_0
    --cache-type-v q4_0
    # Batch sizes: tunable for MoE-on-CPU workload
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    # NUMA: defer to external numactl policy (interleave)
    --numa numactl
    # Metrics endpoint for monitoring
    --metrics
    # Timeout for large context operations
    --timeout 900
)

# GPU-dependent options (differ by backend)
if [[ "$N_GPU_LAYERS" -gt 0 ]]; then
    SERVER_ARGS+=(
        --flash-attn on
    )
    if [[ "$LLAMA_BACKEND" == "ik" ]]; then
        # ik_llama.cpp: --cpu-moe keeps MoE expert weights on CPU, attention on GPU
        SERVER_ARGS+=( --cpu-moe )
        echo "[$(date)] GPU layers: $N_GPU_LAYERS (ik_llama.cpp --cpu-moe, flash-attn on)"
    else
        # mainline: -ot flag offloads MoE expert FFN layers to CPU
        SERVER_ARGS+=( -ot ".ffn_.*_exps.=CPU" )
        echo "[$(date)] GPU layers: $N_GPU_LAYERS (mainline, MoE experts offloaded to CPU, flash-attn on)"
    fi
else
    echo "[$(date)] CPU-only mode (N_GPU_LAYERS=0) — all computation on CPU"
fi

# Conditionally add --jinja and --chat-template-kwargs
if [[ "$JINJA_MODE" == "on" ]]; then
    SERVER_ARGS+=(
        --jinja
        --chat-template-kwargs '{"enable_thinking": false}'
    )
    echo "[$(date)] Jinja template processing: ENABLED (--jinja)"
else
    echo "[$(date)] Jinja template processing: DISABLED (JINJA_MODE=$JINJA_MODE)"
fi

# Run with NUMA interleave for maximum aggregate memory bandwidth.
# MoE expert FFN layers (~196GB) are CPU-bound — distributing pages across all
# 4 NUMA domains gives ~4x bandwidth vs binding to GPU-local node.
echo "[$(date)] Launching with NUMA interleave (all nodes) for MoE bandwidth"

if [[ "$LLAMA_BACKEND" == "ik" ]]; then
    # ik_llama.cpp: uses wrapper script that sets up conda glibc 2.34 + libs
    echo "[$(date)] Using ik_llama.cpp backend: $IK_LLAMA_BIN/run-ik-server.sh"
    numactl --interleave=all \
        "$IK_LLAMA_BIN/run-ik-server.sh" \
        "${SERVER_ARGS[@]}" &
else
    # mainline: direct binary with CUDA libs
    # For CPU-only mode, hide CUDA devices to prevent ggml-cuda assertions on MoE routing
    CUDA_ENV=""
    if [[ "$N_GPU_LAYERS" -eq 0 ]]; then
        CUDA_ENV="CUDA_VISIBLE_DEVICES="
        echo "[$(date)] CUDA_VISIBLE_DEVICES unset for CPU-only mode"
    fi
    env $CUDA_ENV \
    LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    numactl --interleave=all \
        "$LLAMA_BIN/llama-server" \
        "${SERVER_ARGS[@]}" &
fi
SERVER_PID=$!

echo "[$(date)] llama-server started (PID $SERVER_PID)"

###############################################################################
# Wait for llama-server health, then launch chat proxy
###############################################################################

echo "[$(date)] Waiting for llama-server to become healthy..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] llama-server healthy after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date)] ERROR: llama-server exited during startup"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "[$(date)] WARNING: llama-server not healthy after 120s — starting proxy anyway"
fi

# Launch chat-to-completion proxy sidecar
if [[ -f "$PROXY_SCRIPT" ]]; then
    echo "[$(date)] Starting chat proxy on port ${PROXY_PORT} → backend :${PORT}..."
    BACKEND_URL="http://localhost:${PORT}" \
    PROXY_PORT="$PROXY_PORT" \
    python3.11 "$PROXY_SCRIPT" &
    PROXY_PID=$!
    echo "[$(date)] Chat proxy started (PID $PROXY_PID)"

    # Brief wait to verify proxy started
    sleep 2
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "[$(date)] WARNING: Chat proxy failed to start — continuing without proxy"
        PROXY_PID=""
    fi
else
    echo "[$(date)] WARNING: Proxy script not found at $PROXY_SCRIPT — skipping proxy"
    echo "[$(date)] Clients must use /v1/completions directly (chat/completions will hit EOS bug)"
fi

# Wait for server — keeps the job alive and enables graceful shutdown
wait "$SERVER_PID"
