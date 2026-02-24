#!/bin/bash
#SBATCH --job-name=llama-72b
#SBATCH --partition=gpu_ai
#SBATCH --qos=ai_opportunistic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=vasp-01,vasp-03
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --output=/scratch/ai/logs/llama-72b-%j.log
#SBATCH --error=/scratch/ai/logs/llama-72b-%j.err

###############################################################################
# Distributed 72B Inference Server (OR1-Behemoth Q4_K_M)
#
# Purpose: Complex architecture decisions, reasoning tasks
# Throughput: 6-12 tok/s, 2-4s TTFT
# Context: 128K tokens (q4_0 KV cache ~10GB + ~40GB model = ~50GB in 64GB VRAM)
#
# Uses llama.cpp RPC with LAYER SPLIT across 2x V100S GPUs (64GB total).
# 2+1 topology: vasp-01+vasp-03 for 72B, vasp-02 dedicated to 14B fast tier.
# Layer split distributes whole layers equally (1,1 ratio) instead of tensor
# slices, bypassing the head divisibility requirement (64 heads ÷ 2 for TP).
#
# Optimizations applied:
# - NUMA interleave: memory distributed across all 4 vNUMA nodes for bandwidth
# - Matched batch sizes: ubatch=batch=512 reduces network round trips
# - Memory locking: --mlock prevents model swap during long inference
# - q4_0 KV cache: Reduces KV memory footprint in 64GB total VRAM
#
# Note: llama.cpp RPC uses TCP sockets over IPoIB, not native InfiniBand RDMA
# Note: vasp-01 is now dedicated to 72B (Qwen3-Coder-Next moved to vasp-02 router)
#
# This job is preemptible - VASP jobs have priority.
###############################################################################

set -euo pipefail

# Configuration
MODEL="${MODEL:-/scratch/ai/models/or1-behemoth-q4_k_m.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/cluster/shared/llama-cpp/bin}"
CUDA_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/targets/x86_64-linux/lib"
CUDA_COMPAT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/compat"
HEAD_PORT="${HEAD_PORT:-8081}"  # Port 8081 for reasoning tier (8080 = fast tier)
RPC_PORT="${RPC_PORT:-50052}"
CTX_SIZE="${CTX_SIZE:-131072}"
N_PREDICT="${N_PREDICT:--1}"  # -1 = unlimited (default). Reasoning models need full context for CoT.
THREADS="${THREADS:-8}"

# Architectural configuration
# FIX: Increased intervals to handle slow model loading on RPC workers
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-120}" # seconds between health checks (was 60)
MIN_WORKERS="${MIN_WORKERS:-1}"                        # minimum RPC workers for 2-GPU layer split (1 worker required)
WORKER_START_TIMEOUT="${WORKER_START_TIMEOUT:-120}"    # seconds to wait for each worker (was 30)

# Ensure log directory exists
mkdir -p /scratch/ai/logs

# Log startup
echo "=========================================="
echo "llama-72b distributed inference starting"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "Model: ${MODEL}"
echo "Head Port: ${HEAD_PORT}"
echo "RPC Port: ${RPC_PORT}"
echo "Context Size: ${CTX_SIZE}"
echo "=========================================="

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

    # Signal srun-launched RPC workers (SLURM propagates to cgroup)
    for pid in "${RPC_PIDS[@]}"; do
        echo "[$(date)] Signaling srun worker (PID $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Wait for all background processes to complete gracefully
    # This is CRITICAL - allows workers to handle SIGTERM before we exit
    echo "[$(date)] Waiting for processes to complete gracefully..."
    wait

    # Remove endpoint file
    rm -f "$ENDPOINT_FILE" 2>/dev/null || true

    echo "[$(date)] Graceful shutdown complete"
}
trap cleanup SIGTERM SIGINT

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

# Check if a single RPC worker process is still running
# NOTE: TCP probe doesn't work after startup because llama.cpp RPC uses persistent connections.
# Once the head node connects, the worker stops accepting new connections on the port.
# Instead, we check if the srun process (tracked in RPC_PIDS) is still running.
check_worker_health() {
    local node=$1
    local idx=$2  # Index into RPC_PIDS array
    local pid="${RPC_PIDS[$idx]:-}"

    # Check if the srun process is still running
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
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
            echo "[$(date)] Initiating graceful shutdown due to insufficient workers"
            cleanup
            exit 1
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
    exit 1
fi

# Prefetch model on all nodes (parallel)
echo "[$(date)] Prefetching model to page cache..."
for node in "${NODES[@]}"; do
    run_on_node "$node" "numactl --interleave=all sh -c 'vmtouch -vt \"$MODEL\" 2>/dev/null || cat \"$MODEL\" > /dev/null'" &
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

# Start RPC workers on non-head nodes using srun for proper SLURM lifecycle management
# Architecture: Each srun step gets exclusive access to its allocated resources

# FIX: Write logs to shared storage so we can debug crashes
SHARED_LOG_DIR="/cluster/shared/ai/logs/${SLURM_JOB_ID}"
mkdir -p "$SHARED_LOG_DIR"

echo "[$(date)] Starting RPC workers..."
for i in "${!RPC_NODES[@]}"; do
    node="${RPC_NODES[$i]}"
    step_log="${SHARED_LOG_DIR}/rpc-${node}.log"
    echo "  Starting RPC step $i on $node:$RPC_PORT (Logs: $step_log)"

    # srun flags for proper resource isolation:
    # --exclusive: Don't share resources with other steps
    # --gres=gpu:1: Bind exactly one GPU to this step
    # --cpus-per-task: Inherit from job allocation
    # --label: Prefix output with step ID for debugging
    srun --nodes=1 \
         --ntasks=1 \
         --nodelist="$node" \
         --exclusive \
         --gres=gpu:v100s:1 \
         --label \
         --output="$step_log" \
         --error="$step_log" \
        bash -c "numactl --interleave=all LD_LIBRARY_PATH='$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB' CUDA_LAUNCH_BLOCKING=0 '$LLAMA_BIN/rpc-server' --host 0.0.0.0 --port '$RPC_PORT'" &
    RPC_PIDS+=($!)
    WORKER_STATUS[$node]="starting"
    echo "    Step $i PID: ${RPC_PIDS[-1]}"
done

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
    echo "[$(date)] WARNING: No RPC workers available - running in single-GPU mode"
else
    echo "[$(date)] RPC servers: $RPC_SERVERS"
fi

# Register endpoint on shared filesystem (for service discovery)
ENDPOINT_DIR="/cluster/shared/ai/endpoints"
ENDPOINT_FILE="${ENDPOINT_DIR}/${SLURM_JOB_ID}-72b.json"
mkdir -p "${ENDPOINT_DIR}"
cat > "$ENDPOINT_FILE" <<EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "model": "OR1-Behemoth",
    "tier": "reasoning",
    "head_node": "${HEAD_NODE}",
    "rpc_workers": [$(printf '"%s",' "${RPC_NODES[@]}" | sed 's/,$//')],
    "host": "${HEAD_NODE}",
    "port": ${HEAD_PORT},
    "endpoint": "http://${HEAD_NODE}:${HEAD_PORT}/v1/chat/completions",
    "started_at": "$(date -Iseconds)"
}
EOF
echo "[$(date)] Endpoint registered: $ENDPOINT_FILE"

# Start background health monitor
health_monitor &
MONITOR_PID=$!
echo "[$(date)] Health monitor started (PID $MONITOR_PID)"

echo "[$(date)] Starting head node llama-server..."

# Build server command (conditionally include --rpc if workers are available)
SERVER_ARGS=(
    --model "$MODEL"
    --host 0.0.0.0
    --port "$HEAD_PORT"
    --ctx-size "$CTX_SIZE"
    --n-predict "$N_PREDICT"
    # YaRN RoPE scaling for extended context (base model trained at 32K)
    # GGUF has context_length=40960 with no YaRN metadata, so we override both:
    # 1. context_length → 131072 to prevent slot capping at 40960
    # 2. rope-scaling yarn with orig-ctx 32768 for proper frequency interpolation
    --override-kv "qwen3.context_length=int:131072"
    --rope-scaling yarn
    --yarn-orig-ctx 32768
    --split-mode layer
    --tensor-split 1,1
    --n-gpu-layers 80
    --threads "$THREADS"
    --cont-batching
    --metrics
    # Quantized KV cache (q4_0) - fits in VRAM with model across 64GB total
    --cache-type-k q4_0
    --cache-type-v q4_0
    # Batch sizes optimized for network round-trip efficiency
    --batch-size 512
    --ubatch-size 512
    # Lock model in memory to prevent swapping during long inference
    --mlock
    # Single slot for maximum context per request
    --parallel 1
    # Increase timeout for large context operations
    --timeout 600
)

# Only add --rpc if we have alive workers
if [[ -n "$RPC_SERVERS" ]]; then
    SERVER_ARGS+=(--rpc "$RPC_SERVERS")
fi

# Run head node server with NUMA interleave for maximum aggregate memory bandwidth.
# With 4 vNUMA nodes, interleave distributes model weight pages across all domains
# for ~4x bandwidth vs single-node binding. GPU DMA routes through NUMA 0 regardless.
echo "[$(date)] Launching with NUMA interleave (all nodes) for memory bandwidth"
LD_LIBRARY_PATH="$CUDA_COMPAT:$LLAMA_BIN:$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
numactl --interleave=all \
    "$LLAMA_BIN/llama-server" \
    "${SERVER_ARGS[@]}" &
HEAD_PID=$!

echo "[$(date)] Head node server started (PID $HEAD_PID)"

# Wait for all processes - this keeps the job alive and enables graceful shutdown
wait
