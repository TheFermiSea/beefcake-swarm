#!/bin/bash
# Start MiniMax-M2.7 on vasp-03:8081 (GPU-attention + CPU-experts).
#
# Partial-offload layout:
#   • attention, norms, embed, router  → GPU (V100S 32GB, ~15-17 GB)
#   • 256 experts (all FFN blocks)     → CPU (DDR4 256 GB, ~110 GB @ Q3)
#   • KV cache (q8_0)                  → GPU (~6 GB @ 32K ctx)
#
# This layout was adopted 2026-04-18 when we liberated vasp-03's V100
# (GLM-4.7-Flash moved to vasp-03:8082 CPU) specifically so MiniMax's
# attention layers could get GPU acceleration. Pure-CPU MiniMax was
# ~2 tok/s (DDR4 bandwidth-bound); partial-offload targets ~5-10 tok/s
# because attention moves off DDR4 to HBM2.
#
# Binary: llama-server-turboquant (TheTom/llama-cpp-turboquant fork).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-turboquant}"
LOG_PATH="${LOG_PATH:-/tmp/llama-minimax-gpu-offload.log}"
QUANT="${QUANT:-UD-Q3_K_M}"
MODEL_FILE="${MODEL_FILE:-/scratch/ai/models/${QUANT}/MiniMax-M2.7-${QUANT}-00001-of-00004.gguf}"
PORT="${PORT:-8081}"

CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server-turboquant not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk -v port="--port ${PORT}" '$0 ~ "llama-server.*" port "( |$)" { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 5
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "rsync from vasp-01: rsync -a root@vasp-01:/scratch/ai/models/${QUANT}/ /scratch/ai/models/${QUANT}/" >&2
  exit 1
fi

# --n-gpu-layers 999: offload ALL layers to GPU, THEN …
# --override-tensor 'ffn_.*_exps\.=CPU': pull expert FFN tensors back to CPU.
# This keeps attention on GPU (what we want to accelerate) while 256 MoE
# experts remain RAM-resident (bandwidth-bound either way, so no loss).
nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" numactl --interleave=all \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias MiniMax-M2.7 \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 999 \
  --override-tensor 'ffn_.*_exps\.=CPU' \
  --threads 32 --batch-size 512 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --no-mmap --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started MiniMax-M2.7 PID=$! port=${PORT} binary=${LLAMA_SERVER}"
echo "Expected: attention on GPU (~15GB VRAM), experts on CPU (~110GB RAM)"
echo "Target throughput: 5-10 tok/s output (vs ~2 tok/s pure-CPU baseline)"
