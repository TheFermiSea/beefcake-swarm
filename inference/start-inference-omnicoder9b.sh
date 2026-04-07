#!/bin/bash
# Start OmniCoder-9B on vasp-02 port 8083.
# Secondary model: shares GPU with Devstral-24B on port 8081.
# Q4_K_M ~5.4GB fits alongside Devstral (~13GB) in V100S 32GB.
# 16K context, 8 threads (secondary model gets fewer).
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-omnicoder9b.log}"
MODEL_FILE="/scratch/ai/models/OmniCoder-9B-Q4_K_M.gguf"
PORT="${OMNICODER_PORT:-8083}"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'( |$)/ { print $1 }')"
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
  --alias OmniCoder-9B \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 16384 --n-gpu-layers 999 \
  --threads 8 --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started OmniCoder-9B PID=$! port=${PORT} container=${CONTAINER}"
