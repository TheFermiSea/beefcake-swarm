#!/bin/bash
# Start Qwen3.6-35B-A3B MoE on vasp-01:8081 (GPU).
#
# 35B total / 3B active per token. All weights GPU-resident (A3B experts
# are tiny — CPU-offload would only slow this down on a 32GB V100S).
# Q4_K_M ~20.6GB weights + turbo4 KV at 64K ctx ~3-4GB = ~25GB, fits V100S 32GB.
#
# Binary: llama-server-turboquant (TheTom/llama-cpp-turboquant fork).
# Cache type `turbo4` is the TurboQuant 4.125-bpw KV with native CUDA kernels
# on V100 sm_70. Saves ~2-3GB VRAM vs q8_0 at same context length.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-turboquant}"
LOG_PATH="${LOG_PATH:-/tmp/llama-qwen36-turbo.log}"
QUANT="${QUANT:-UD-Q4_K_M}"
MODEL_FILE="/scratch/ai/models/Qwen3.6-35B-A3B-${QUANT}.gguf"

# CUDA runtime (nvhpc_sdk 24.11 ships CUDA 12.6)
CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server-turboquant not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port 8081( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 3
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  exit 1
fi

nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" numactl --interleave=all \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen3.6-35B-A3B (${QUANT}) PID=$! binary=${LLAMA_SERVER}"
echo "Client-side sampling (coding, thinking off): temperature=0.7, top_p=0.8,"
echo "  top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
