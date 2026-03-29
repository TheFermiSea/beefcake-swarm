#!/bin/bash
# Start Qwen2.5-Coder-14B as secondary model on vasp-03 port 8083.
# Runs alongside OmniCoder-9B (port 8081) — both fit in 32GB VRAM.
# Combined: 5.7GB + 8.5GB = ~14.2GB, leaving ~18GB for KV caches.
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-coder14b.log}"
MODEL_FILE="/scratch/ai/models/Qwen2.5-Coder-14B-Q4_K_M.gguf"
PORT="${CODER14B_PORT:-8083}"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

# Kill only processes on our port, not the primary model on 8081
existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  exit 1
fi

nohup numactl --interleave=all apptainer run --nv --bind /scratch/ai:/scratch/ai:ro "${CONTAINER}" \
  --model "${MODEL_FILE}" \
  --alias Qwen2.5-Coder-14B \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 16384 --n-gpu-layers 999 \
  --threads 16 --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen2.5-Coder-14B PID=$! port=${PORT} container=${CONTAINER}"
