#!/bin/bash
# Start Devstral Small 2 24B (Mistral agentic coding model) on vasp-02.
# Dense model: all weights on GPU, no expert offload needed.
# Q4_K_M ~14.3GB fits in V100S 32GB with ~17GB for KV cache.
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"
MODEL_FILE="/scratch/ai/models/Devstral-Small-2-24B-Q4_K_M.gguf"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

existing_pids="$(ps -eo pid=,args= | awk '/(llama-server-mmq|apptainer .*llama-server\.sif|\/usr\/local\/bin\/llama-server)( |$)/ { print $1 }')"
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
  --alias Devstral-Small-2-24B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started Devstral-Small-2-24B PID=$! container=${CONTAINER}"
