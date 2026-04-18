#!/bin/bash
# Start Gemma-4-31B-it on vasp-02:8081 (GPU).
# Dense 30.7B model, Q4_K_M ~18.3GB — fits V100S 32GB with 10-12GB KV headroom.
# Native 256K context; capped at 32K for Q4 KV cache.
#
# IMPORTANT: Gemma-4 has a reasoning channel (<|channel>thought…<channel|>)
# which llama.cpp parses as hidden CoT, leaving the visible `content` field
# empty. Always pass --reasoning off for this deployment — we use mini-SWE-agent
# which parses visible text, not reasoning traces.
#
# REQUIRES: llama-server binary built from llama.cpp >= 2026-04-12, including
# PRs #21534 (gemma4 tokenizer UTF-8 fix), #21625 (multimodal padding),
# #21661 (grammar rule), #21697 (reasoning budget sampler).
# Build on vasp-02 with gcc-toolset-13 + full V100S flags; default install at
# /usr/local/bin/llama-server-latest-20260418 (see inference/build-llama-cpp.sh
# if we ever script the build).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-latest-20260418}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-gemma4-31b.log}"
MODEL_FILE="/scratch/ai/models/google_gemma-4-31B-it-Q4_K_M.gguf"
PORT="${GEMMA_PORT:-8081}"

# CUDA runtime libraries (hpc_sdk 24.11 bundles CUDA 12.6)
CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: Gemma4-capable llama-server not found: ${LLAMA_SERVER}" >&2
  echo "Build latest llama.cpp with gcc-toolset-13 and install here." >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  echo "WARNING: Killing existing llama-server on port ${PORT}"
  kill ${existing_pids} 2>/dev/null || true
  sleep 5
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "Download: huggingface-cli download bartowski/google_gemma-4-31B-it-GGUF 'google_gemma-4-31B-it-Q4_K_M.gguf' --local-dir /scratch/ai/models/" >&2
  exit 1
fi

nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Gemma-4-31B-it \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 16 --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -fa on --parallel 1 --mlock --cont-batching --metrics \
  --jinja --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started Gemma-4-31B-it PID=$! port=${PORT} binary=${LLAMA_SERVER}"
echo "Expected warm-up: ~15-20s to load ${MODEL_FILE} (~18GB)"
