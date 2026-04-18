#!/bin/bash
# Start GLM-4.7-Flash (30B/3B MoE, deepseek2-style MLA) on vasp-03:8081 (GPU).
# Native OpenAI-format tool calling, 200K context window.
#
# Binary: llama-server-turboquant (TheTom fork).
# KV cache: q8_0 — NOT turbo4. GLM's deepseek2 architecture uses MLA
# with 576/512/256 head dims (key_length/value_length/key_length_mla);
# turbo4 kernels in the fork are specialized for standard transformer
# head dims and crash during load with deepseek2. q8_0 is the safe
# compromise on V100 (same VRAM footprint would be ~4 GB at 32K ctx).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-turboquant}"
LOG_PATH="${LOG_PATH:-/tmp/llama-glm-flash.log}"
MODEL_FILE="/scratch/ai/models/GLM-4.7-Flash-Q4_K_M.gguf"

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
  --alias GLM-4.7-Flash \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started GLM-4.7-Flash PID=$! binary=${LLAMA_SERVER}"
