#!/bin/bash
# Start GLM-4.7-Flash (30B/3B MoE, SOTA tool-calling) on vasp-03.
# 18.3GB Q4_K_M fits in V100S 32GB. Native OpenAI-format tool calling.
# 200K context window, 3B active params → fast inference.
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"
MODEL_FILE="/scratch/ai/models/GLM-4.7-Flash-Q4_K_M.gguf"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

existing_pids="$(ps -eo pid=,args= | awk '/([Aa]pptainer .*llama-server\.sif|llama-server-mmq|\/usr\/local\/bin\/llama-server)( |$)/ { print $1 }')"
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
  --alias GLM-4.7-Flash \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started GLM-4.7-Flash PID=$! container=${CONTAINER}"
