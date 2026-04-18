#!/bin/bash
# Start Gemma-4-31B-it on vasp-02:8081 (GPU).
# Dense 30.7B, Q4_K_M ~18.3GB + turbo4 KV at 32K ctx ~2-3GB ≈ 21-22GB / 32GB V100S.
#
# Binary: llama-server-turboquant (TheTom/llama-cpp-turboquant fork) — has
# native CUDA TurboQuant kernels + all the Gemma-4 fixes through 2026-04.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-turboquant}"
LOG_PATH="${LOG_PATH:-/tmp/llama-gemma4-turbo.log}"
MODEL_FILE="/scratch/ai/models/google_gemma-4-31B-it-Q4_K_M.gguf"
PORT="${GEMMA_PORT:-8081}"

CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server-turboquant not found: ${LLAMA_SERVER}" >&2
  echo "Build instructions: clone https://github.com/TheTom/llama-cpp-turboquant" >&2
  echo "  checkout feature/turboquant-kv-cache branch, gcc-toolset-13 + V100 CMake flags." >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  echo "Killing existing llama-server on port ${PORT}"
  kill ${existing_pids} 2>/dev/null || true
  sleep 5
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  exit 1
fi

# --reasoning off: Gemma-4's <|channel>thought…<channel|> routes to
# reasoning_content; chat response `content` stays clean for the
# mini-SWE-agent text parser.
nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Gemma-4-31B-it \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 16 --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  -fa on --parallel 1 \
  --mlock --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started Gemma-4-31B-it PID=$! port=${PORT} binary=${LLAMA_SERVER}"
echo "Client-side sampling (Google card): temperature=1.0, top_p=0.95, top_k=64"
