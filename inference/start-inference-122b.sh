#!/bin/bash
# Start Qwen3.5-122B-A10B from the shared Apptainer image on vasp-01/02.
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache

existing_pids="$(ps -eo pid=,args= | awk '/(llama-server-mmq|apptainer .*llama-server\.sif|\/usr\/local\/bin\/llama-server)( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

nohup numactl --interleave=all apptainer run --nv --bind /scratch/ai:/scratch/ai:ro "${CONTAINER}" \
  --model /scratch/ai/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  --alias Qwen3.5-122B-A10B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 16384 --n-gpu-layers 99 \
  -ot ".ffn_.*_exps.=CPU" \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen3.5-122B-A10B PID=$! container=${CONTAINER}"
