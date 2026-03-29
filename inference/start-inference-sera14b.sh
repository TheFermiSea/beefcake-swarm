#!/bin/bash
# Start SERA-14B (Allen AI SWE agent, Qwen3 backbone) on vasp-03 port 8083.
# Secondary model: shares GPU with GLM-4.7-Flash on port 8081.
# Q4_K_M ~9GB fits alongside GLM-4.7-Flash in V100S 32GB.
# 8K context (VRAM-constrained), 8 threads (secondary model gets fewer).
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-sera14b.log}"
MODEL_FILE="/scratch/ai/models/SERA-14B-Q4_K_M.gguf"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port 8083( |$)/ { print $1 }')"
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
  --alias SERA-14B \
  --host 0.0.0.0 --port 8083 \
  --ctx-size 8192 --n-gpu-layers 999 \
  --threads 8 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started SERA-14B PID=$! container=${CONTAINER}"
