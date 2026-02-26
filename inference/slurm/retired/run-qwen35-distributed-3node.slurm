#!/bin/bash
#SBATCH --job-name=llama-qwen35
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=vasp-01,vasp-02,vasp-03
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-qwen35-%j.log
#SBATCH --error=/scratch/ai/logs/llama-qwen35-%j.err

###############################################################################
# Distributed Qwen3.5-397B-A17B Inference Server (Q4_K_XL MoE)
#
# Purpose: Local reasoning/manager model — replaces OR1-Behemoth.
#   Serves as council Strategist + local manager fallback.
#
# Model: Qwen3.5-397B-A17B-UD-Q4_K_XL (~214GB, 397B total / 17B active MoE)
# Throughput: ~5 tok/s gen, ~15 tok/s prompt (single), ~6.3 agg gen (4 concurrent)
# Context: 256K tokens native (no YaRN needed), 4 parallel slots default
#
# Uses llama.cpp RPC with LAYER SPLIT across 3x V100S GPUs (96GB total VRAM).
# MoE expert FFN layers are offloaded to CPU (-ot ".ffn_.*_exps.=CPU").
#
# Memory budget (per node, 256GB RAM):
#   Model weights: ~214GB / 3 ≈ 72GB (GPU layers + CPU MoE)
#   KV cache: 4 slots × ~16GB q4_0 / 3 ≈ 22GB per node
#   OS + headroom: ~16GB
#   Total: ~110GB per node of 256GB — plenty of room for 6-8 slots
#
# CPU allocation (36 cores/node):
#   MoE expert computation dominates — all 36 threads available
#   --threads controls decode (generation) threads, --threads-batch controls prompt/batch
#   Decode can saturate memory bandwidth with fewer threads than batch processing
#
# NUMA topology (4 vNUMA nodes per VM, matching host SNC):
#   vNUMA 0-1: Physical Socket 0 (GPU-local), ~64GB each
#   vNUMA 2-3: Physical Socket 1, ~64GB each
#   Memory interleaved across all 4 NUMA nodes (numactl --interleave=all)
#   for maximum aggregate bandwidth on CPU-bound MoE expert computation.
#   GPU DMA routes through NUMA 0 regardless of interleave policy.
#   --numa numactl tells llama.cpp to defer NUMA policy to external numactl.
#
# Scaling: Set PARALLEL_SLOTS=6 or PARALLEL_SLOTS=8 at launch time.
#   8 slots: ~342GB total across 768GB RAM — fits with 400+ GB headroom.
#
# This job is preemptible - VASP jobs have priority.
###############################################################################

set -euo pipefail

# Configuration
# Split GGUF: llama.cpp loads all shards when given the first shard path
MODEL="${MODEL:-/scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
HEAD_PORT="${HEAD_PORT:-8081}"  # Port 8081 for manager tier (replaces OR1-Behemoth)
RPC_PORT="${RPC_PORT:-50052}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-4}"
CTX_SIZE="${CTX_SIZE:-262144}"  # 256K native context per slot
N_PREDICT="${N_PREDICT:--1}"    # -1 = unlimited (default). Reasoning models need full context for CoT.
THREADS="${THREADS:-28}"             # Decode threads — fewer than max reduces memory bandwidth contention
THREADS_BATCH="${THREADS_BATCH:-36}" # Batch/prompt threads — use all cores for prefill throughput
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

# Architectural configuration
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-120}" # seconds between health checks
MIN_WORKERS="${MIN_WORKERS:-2}"                        # minimum RPC workers for 3-GPU layer split
WORKER_START_TIMEOUT="${WORKER_START_TIMEOUT:-180}"    # seconds to wait for each worker (model is large)

# Ensure log directory exists
mkdir -p /scratch/ai/logs

# Log startup
echo "=========================================="
echo "llama-qwen35 distributed inference starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "Model: ${MODEL}"
echo "Head Port: ${HEAD_PORT}"
echo "RPC Port: ${RPC_PORT}"
echo "Parallel Slots: ${PARALLEL_SLOTS}"
echo "Context Size: ${CTX_SIZE} (per slot)"
echo "Threads: ${THREADS}"
echo "Threads (batch): ${THREADS_BATCH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Ubatch size: ${UBATCH_SIZE}"
echo "=========================================="

###############################################################################
# Memory Validation
###############################################################################

MODEL_SIZE_GB=214
KV_PER_SLOT_GB=16  # q4_0 KV cache for 256K context
OS_HEADROOM_GB=16
NUM_NODES=3

TOTAL_KV_GB=$((KV_PER_SLOT_GB * PARALLEL_SLOTS))
TOTAL_MEMORY_GB=$((MODEL_SIZE_GB + TOTAL_KV_GB))
PER_NODE_GB=$(( (TOTAL_MEMORY_GB + NUM_NODES - 1) / NUM_NODES + OS_HEADROOM_GB ))

echo "[$(date)] Memory estimate: ${MODEL_SIZE_GB}GB model + ${TOTAL_KV_GB}GB KV (${PARALLEL_SLOTS} slots) = ${TOTAL_MEMORY_GB}GB total"
echo "[$(date)] Per-node estimate: ~${PER_NODE_GB}GB (including ${OS_HEADROOM_GB}GB OS headroom)"

if [[ $PER_NODE_GB -gt 240 ]]; then
    echo "[$(date)] WARNING: Per-node memory estimate (${PER_NODE_GB}GB) exceeds allocation (240GB)"
    echo "[$(date)] Reduce PARALLEL_SLOTS or increase --mem. Current: PARALLEL_SLOTS=${PARALLEL_SLOTS}"
    echo "[$(date)] Formula: (${MODEL_SIZE_GB} + ${KV_PER_SLOT_GB} * SLOTS) / ${NUM_NODES} + ${OS_HEADROOM_GB}"
    exit 1
fi

# Parse node list (SC2207-compliant array assignment)
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
RPC_NODES=("${NODES[@]:1}")

echo "Head node: $HEAD_NODE"
echo "RPC workers: ${RPC_NODES[*]}"

# Graceful shutdown handler
# NOTE: srun processes are in SLURM cgroup and auto-cleaned on job end
# We only need to signal our tracked PIDs and wait for graceful completion
RPC_PIDS=()
HEAD_PID=""
MONITOR_PID=""
SHUTTING_DOWN=false
ENDPOINT_FILE=""

cleanup() {
    # Prevent re-entry
    if $SHUTTING_DOWN; then
        return
    fi
    SHUTTING_DOWN=true

    echo "[$(date)] SIGTERM received - initiating graceful shutdown"

    # Stop health monitor first
    if [[ -n "$MONITOR_PID" ]]; then
        echo "[$(date)] Stopping health monitor (PID $MONITOR_PID)..."
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
    fi

    # Signal head node server (if running)
    if [[ -n "$HEAD_PID" ]]; then
        echo "[$(date)] Signaling head node server (PID $HEAD_PID)..."
        kill -TERM "$HEAD_PID" 2>/dev/null || true
    fi

    # Kill SSH-launched RPC workers on remote nodes
    for node in "${RPC_NODES[@]}"; do
        echo "[$(date)] Killing rpc-server on $node..."
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" \
            "pkill -f 'rpc-server.*--port $RPC_PORT'" 2>/dev/null || true
    done

    # Wait for all background processes to complete gracefully
    # This is CRITICAL - allows workers to handle SIGTERM before we exit
    echo "[$(date)] Waiting for processes to complete gracefully..."
    wait

    # Remove endpoint file
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true

    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup EXIT

###############################################################################
# Health Monitoring & Graceful Degradation
###############################################################################

# Track worker status: node -> "alive" | "dead" | "unknown"
declare -A WORKER_STATUS

# TCP probe for startup - checks if RPC worker port is listening
# Used ONLY during initial startup before the head node connects.
check_worker_port() {
    local node=$1
    if timeout 5 bash -c "echo >/dev/tcp/$node/$RPC_PORT" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if a single RPC worker process is still running on its remote node.
# NOTE: TCP probe doesn't work after startup because llama.cpp RPC uses persistent connections.
# Once the head node connects, the worker stops accepting new connections on the port.
# With SSH-launched workers, we check the remote process via SSH.
check_worker_health() {
    local node=$1
    local idx=$2  # unused but kept for API compat
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" \
        "pgrep -f 'rpc-server.*--port $RPC_PORT'" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Count alive workers
count_alive_workers() {
    local count=0
    for node in "${RPC_NODES[@]}"; do
        if [[ "${WORKER_STATUS[$node]:-unknown}" == "alive" ]]; then
            ((count++)) || true
        fi
    done
    echo "$count"
}

# Background health monitor
# Uses process-based health checking (not TCP probe) because llama.cpp RPC
# holds persistent connections and doesn't accept new connections after init.
health_monitor() {
    echo "[$(date)] Health monitor started (interval: ${HEALTH_CHECK_INTERVAL}s, checking process PIDs)"
    while ! $SHUTTING_DOWN; do
        sleep "$HEALTH_CHECK_INTERVAL"
        if $SHUTTING_DOWN; then
            break
        fi

        local alive_count=0
        for i in "${!RPC_NODES[@]}"; do
            local node="${RPC_NODES[$i]}"
            if check_worker_health "$node" "$i"; then
                if [[ "${WORKER_STATUS[$node]:-unknown}" != "alive" ]]; then
                    echo "[$(date)] Worker $node (PID ${RPC_PIDS[$i]}) is now ALIVE"
                fi
                WORKER_STATUS[$node]="alive"
                ((alive_count++)) || true
            else
                if [[ "${WORKER_STATUS[$node]:-unknown}" == "alive" ]]; then
                    echo "[$(date)] WARNING: Worker $node (PID ${RPC_PIDS[$i]:-?}) is now DEAD"
                fi
                WORKER_STATUS[$node]="dead"
            fi
        done

        # Check degradation threshold
        if [[ $alive_count -lt $MIN_WORKERS ]]; then
            echo "[$(date)] CRITICAL: Only $alive_count workers alive (minimum: $MIN_WORKERS)"
            echo "[$(date)] Signaling parent (PID $$) for graceful shutdown"
            kill -TERM $$ 2>/dev/null || true
            return
        fi
    done
    echo "[$(date)] Health monitor stopped"
}

###############################################################################
# Model Distribution Functions (SSH for utility commands, srun for workers)
###############################################################################

# Check model on a node (SSH for remote to avoid nested srun reliability issues)
check_model() {
    local node=$1
    if [[ "$node" == "$(hostname -s)" || "$node" == "$(hostname -f)" ]]; then
        [[ -f "$MODEL" ]]
    else
        ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "[[ -f '$MODEL' ]]" 2>/dev/null
    fi
}

# Run command on node (SSH for remote utility commands)
run_on_node() {
    local node=$1
    shift
    if [[ "$node" == "$(hostname -s)" || "$node" == "$(hostname -f)" ]]; then
        bash -c "$*"
    else
        ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "$*" 2>/dev/null
    fi
}

# Ensure model is available on all nodes
echo "[$(date)] Checking model availability..."
MISSING_NODES=()
for node in "${NODES[@]}"; do
    echo -n "  $node: "
    if check_model "$node"; then
        echo "OK"
    else
        echo "MISSING"
        MISSING_NODES+=("$node")
    fi
done

if [[ ${#MISSING_NODES[@]} -gt 0 ]]; then
    echo "[$(date)] ERROR: Model missing on: ${MISSING_NODES[*]}"
    echo "Please ensure $MODEL exists on all nodes."
    echo "Model is stored on local /scratch (not NFS), so it must be copied to each node."
    echo ""
    echo "Download once and copy to other nodes:"
    echo "  # Download to one node first (split GGUF, 6 shards):"
    echo "  ssh root@10.0.0.20 'python3.11 -c \"from huggingface_hub import snapshot_download;"
    echo "    snapshot_download(\\\"unsloth/Qwen3.5-397B-A17B-GGUF\\\","
    echo "      local_dir=\\\"/scratch/ai/models/Qwen3.5-397B-A17B-GGUF\\\","
    echo "      allow_patterns=[\\\"UD-Q4_K_XL/*\\\"])\"'"
    echo ""
    echo "  # Then copy to the other nodes:"
    echo "  for dest in 10.0.0.21 10.0.0.22; do"
    echo "    rsync -avP root@10.0.0.20:/scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/ \\"
    echo "      root@\$dest:/scratch/ai/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/"
    echo "  done"
    exit 1
fi

# Prefetch model to page cache on all nodes (parallel)
# NOTE: No --mlock — model is ~214GB, too large to lock. vmtouch prefetch is sufficient.
echo "[$(date)] Prefetching model to page cache on all nodes..."
MODEL_DIR="$(dirname "$MODEL")"
for node in "${NODES[@]}"; do
    run_on_node "$node" "numactl --interleave=all sh -c 'vmtouch -vt \"$MODEL_DIR\"/*.gguf 2>/dev/null || cat \"$MODEL_DIR\"/*.gguf > /dev/null'" &
done
wait
echo "[$(date)] Prefetch complete"

# Check for stale RPC processes from previous jobs (informational only)
# NOTE: We don't force-kill here as fuser -k could affect other jobs
# SLURM cgroups ensure proper isolation for srun-launched processes
echo "[$(date)] Checking for port availability..."
for node in "${RPC_NODES[@]}"; do
    if run_on_node "$node" "timeout 2 bash -c 'echo >/dev/tcp/localhost/${RPC_PORT}'" 2>/dev/null; then
        echo "WARNING: Port $RPC_PORT already in use on $node - may conflict"
    fi
done

# Start RPC workers on non-head nodes using SSH for reliable remote launch.
# srun steps within batch jobs can timeout on small clusters; SSH is more reliable
# and the SLURM cgroup still contains the batch allocation for resource limits.

# Write logs to shared storage for debugging
SHARED_LOG_DIR="/cluster/shared/ai/logs/${SLURM_JOB_ID}"
mkdir -p "$SHARED_LOG_DIR"

echo "[$(date)] Starting RPC workers..."
for i in "${!RPC_NODES[@]}"; do
    node="${RPC_NODES[$i]}"
    step_log="${SHARED_LOG_DIR}/rpc-${node}.log"
    echo "  Starting RPC worker $i on $node:$RPC_PORT (Logs: $step_log)"

    ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" \
        "LD_LIBRARY_PATH='$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB' \
         CUDA_LAUNCH_BLOCKING=0 \
         nohup numactl --interleave=all \
         '$LLAMA_BIN/rpc-server' --host 0.0.0.0 --port '$RPC_PORT' \
         > '$step_log' 2>&1 &
         echo \$!" &
    RPC_PIDS+=($!)
    WORKER_STATUS[$node]="starting"
    echo "    SSH launcher PID: ${RPC_PIDS[-1]}"
done
# Wait for SSH commands to complete (they return after launching nohup background process)
sleep 5

# Wait for RPC workers with timeout and validation
# Uses TCP probe here because persistent connection hasn't been established yet
echo "[$(date)] Waiting for RPC workers to initialize (timeout: ${WORKER_START_TIMEOUT}s per worker)..."
WORKERS_READY=0
for node in "${RPC_NODES[@]}"; do
    echo -n "  Checking $node:$RPC_PORT... "
    start_time=$SECONDS
    while ! check_worker_port "$node"; do
        if (( SECONDS - start_time > WORKER_START_TIMEOUT )); then
            echo "TIMEOUT"
            WORKER_STATUS[$node]="dead"
            break
        fi
        sleep 1
    done
    if check_worker_port "$node"; then
        echo "READY"
        WORKER_STATUS[$node]="alive"
        ((WORKERS_READY++)) || true
    fi
done

echo "[$(date)] Workers ready: $WORKERS_READY / ${#RPC_NODES[@]}"

# Check minimum worker threshold
if [[ $WORKERS_READY -lt $MIN_WORKERS ]]; then
    echo "[$(date)] FATAL: Insufficient workers ($WORKERS_READY < $MIN_WORKERS minimum)"
    cleanup
    exit 1
fi

# Build RPC server list for head node (only include alive workers)
RPC_SERVERS=""
for node in "${RPC_NODES[@]}"; do
    if [[ "${WORKER_STATUS[$node]}" == "alive" ]]; then
        if [[ -n "$RPC_SERVERS" ]]; then
            RPC_SERVERS+=","
        fi
        RPC_SERVERS+="${node}:${RPC_PORT}"
    fi
done

if [[ -z "$RPC_SERVERS" ]]; then
    echo "[$(date)] FATAL: No RPC workers available — cannot run distributed inference"
    cleanup
    exit 1
fi
echo "[$(date)] RPC servers: $RPC_SERVERS"

# Register endpoint on shared filesystem (for service discovery)
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-qwen35.json"
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "Qwen3.5-397B-A17B",
    "tier": "manager-local",
    "head_node": "${HEAD_NODE}",
    "rpc_workers": [$(printf '"%s",' "${RPC_NODES[@]}" | sed 's/,$//')],
    "host": "${HEAD_NODE}",
    "port": ${HEAD_PORT},
    "endpoint": "http://${HEAD_NODE}:${HEAD_PORT}/v1/chat/completions",
    "parallel_slots": ${PARALLEL_SLOTS},
    "context_per_slot": ${CTX_SIZE},
    "started_at": "$(date -Iseconds)"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

# Start background health monitor
health_monitor &
MONITOR_PID=$!
echo "[$(date)] Health monitor started (PID $MONITOR_PID)"

echo "[$(date)] Starting head node llama-server..."

# Build server command
SERVER_ARGS=(
    --model "$MODEL"
    --alias "Qwen3.5-397B-A17B"
    --host 0.0.0.0
    --port "$HEAD_PORT"
    --ctx-size "$CTX_SIZE"
    --n-predict "$N_PREDICT"
    # RPC layer split across 3 GPUs (equal distribution)
    --rpc "$RPC_SERVERS"
    --split-mode layer
    --tensor-split 1,1,1
    --n-gpu-layers 99
    # MoE expert FFN offload to CPU — this is the key optimization for 397B MoE.
    # Only the 17B active parameters (attention + routing) stay on GPU.
    # Expert FFN layers (~196GB) run on CPU across all 3 nodes.
    -ot ".ffn_.*_exps.=CPU"
    # Parallel slots for concurrent sessions
    --parallel "$PARALLEL_SLOTS"
    # Threading for MoE CPU computation
    --threads "$THREADS"
    --threads-batch "$THREADS_BATCH"
    # Continuous batching for multi-slot throughput
    --cont-batching
    # Flash attention for memory efficiency
    --flash-attn on
    # Jinja2 chat template with model's native thinking support
    --jinja
    # KNOWN ISSUE (llama.cpp #19690): Qwen3.5's hybrid Mamba/attention architecture
    # forces full prompt re-processing on every request. The recurrent (Mamba) state
    # also causes premature EOS after <think> tokens in chat mode.
    # When the upstream fix lands, thinking will work without config changes.
    # WORKAROUND: Use /completion endpoint with plain text Q&A formatting for now.
    # e.g. "Question: ...\nAnswer: " format produces coherent multi-sentence output.
    # Quantized KV cache — q4_0 keeps 256K context manageable
    --cache-type-k q4_0
    --cache-type-v q4_0
    # Batch sizes: tunable for MoE-on-CPU workload
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    # NUMA: defer to external numactl policy (interleave)
    --numa numactl
    # Metrics endpoint for monitoring
    --metrics
    # Longer timeout for 256K context operations
    --timeout 900
)

# Run head node server with NUMA interleave for maximum aggregate memory bandwidth.
# MoE expert FFN layers (~196GB across 3 nodes) are CPU-bound — distributing pages
# across all 4 NUMA domains gives ~4x the bandwidth vs binding to GPU-local node.
# GPU DMA naturally routes through NUMA 0 regardless of interleave policy.
# NOTE: No --mlock — model is ~214GB, too large to lock in memory.
# vmtouch prefetch above ensures pages are in cache for fast startup.
echo "[$(date)] Launching with NUMA interleave (all nodes) for MoE bandwidth"
LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
numactl --interleave=all \
    "$LLAMA_BIN/llama-server" \
    "${SERVER_ARGS[@]}" &
HEAD_PID=$!

echo "[$(date)] Head node server started (PID $HEAD_PID)"

# Wait for all processes - this keeps the job alive and enables graceful shutdown
wait
