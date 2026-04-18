#!/bin/bash
# Start GLM-4.7-Flash on vasp-03:8082 (CPU).
# 30B MoE / 3B active — sparse MoE runs well on CPU via DDR4 bandwidth.
# Q4_K_M ~17GB weights into RAM. Co-resident with MiniMax-M2.7 (GPU+CPU
# partial-offload) on the same node; combined RAM footprint ~131GB.
#
# Moved from vasp-03:8081 GPU → :8082 CPU on 2026-04-18 to free the GPU
# for MiniMax-M2.7's attention layers.
#
# Binary: llama-server-turboquant.
# KV cache: q8_0 — deepseek2/MLA arch in GLM's GGUF metadata (576/512/256
# head dims) is incompatible with turbo4 kernels; load crashes with turbo4.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-turboquant}"
LOG_PATH="${LOG_PATH:-/tmp/llama-glm-cpu.log}"
MODEL_FILE="/scratch/ai/models/GLM-4.7-Flash-Q4_K_M.gguf"
PORT="${PORT:-8082}"

CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server-turboquant not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk -v port="--port ${PORT}" '$0 ~ "llama-server.*" port "( |$)" { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 3
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  exit 1
fi

# --threads 24: leave ~8 cores for the co-resident MiniMax expert pool.
# (vasp-03 has ~36 logical cores; MiniMax uses 32, GLM uses 24, some
# overlap is fine since they rarely both peak simultaneously on swarm
# dispatch.)
nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" numactl --interleave=all \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias GLM-4.7-Flash \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 0 \
  --threads 24 --batch-size 512 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --no-mmap --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started GLM-4.7-Flash PID=$! port=${PORT} (CPU)"
