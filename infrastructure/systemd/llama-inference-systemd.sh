#!/bin/bash
# Systemd-compatible wrapper for llama-server inference.
# Runs in foreground (no nohup, no &) so systemd can manage lifecycle.
# Reads model config from /opt/beefcake-swarm/inference/model.conf
#
# model.conf format (one VAR=VALUE per line):
#   MODEL_FILE=/scratch/ai/models/Qwen3.5-27B-Q4_K_M.gguf
#   MODEL_ALIAS=Qwen3.5-27B
#   PORT=8081
#   CTX_SIZE=32768
#   THREADS=32
#   BATCH_SIZE=4096
set -euo pipefail

CONF="${1:-/opt/beefcake-swarm/inference/model.conf}"
if [[ ! -f "$CONF" ]]; then
    echo "ERROR: Config not found: $CONF" >&2
    exit 1
fi
source "$CONF"

# Validate required vars
: "${MODEL_FILE:?MODEL_FILE must be set in $CONF}"
: "${MODEL_ALIAS:?MODEL_ALIAS must be set in $CONF}"
: "${PORT:=8081}"
: "${CTX_SIZE:=32768}"
: "${THREADS:=32}"
: "${BATCH_SIZE:=4096}"
: "${UBATCH_SIZE:=$BATCH_SIZE}"
: "${GPU_LAYERS:=999}"

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"

if [[ ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: Model file not found: $MODEL_FILE" >&2
    exit 1
fi

if ! command -v apptainer &>/dev/null; then
    echo "ERROR: apptainer not found" >&2
    exit 1
fi

export HOME=/tmp
export CUDA_CACHE_PATH=/tmp/cuda-cache
mkdir -p /tmp/cuda-cache

echo "Starting $MODEL_ALIAS on port $PORT (ctx=$CTX_SIZE, threads=$THREADS)"

# Run in foreground — systemd manages the lifecycle
exec numactl --interleave=all apptainer run --nv --bind /scratch/ai:/scratch/ai:ro "$CONTAINER" \
    --model "$MODEL_FILE" \
    --alias "$MODEL_ALIAS" \
    --host 0.0.0.0 --port "$PORT" \
    --ctx-size "$CTX_SIZE" --n-gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS" --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja
