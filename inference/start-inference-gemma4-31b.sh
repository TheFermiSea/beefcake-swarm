#!/bin/bash
# Start Gemma-4-31B-it on a dedicated node (port 8081).
# Dense 30.7B model, Q4_K_M ~19.6GB — requires exclusive use of a V100S 32GB.
# Cannot coexist with other large models; must REPLACE the primary on a node.
# 256K native context, but limit to 16K for VRAM headroom with Q8 KV cache.
#
# REQUIRES: llama-server-gemma4 binary (b8692+) with Gemma 4 support.
# The standard container (llama-server.sif, ~b8231) does NOT support Gemma 4.
# Build: see inference/ build notes or /cluster/shared/llama-cpp/bin/.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-gemma4}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-gemma4-31b.log}"
MODEL_FILE="/scratch/ai/models/google_gemma-4-31B-it-Q4_K_M.gguf"
PORT="${GEMMA_PORT:-8081}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: Gemma4-capable llama-server not found: ${LLAMA_SERVER}" >&2
  echo "Build latest llama.cpp (b8637+) and install to ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  echo "WARNING: Killing existing llama-server on port ${PORT}"
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "Download: wget -O '${MODEL_FILE}' 'https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF/resolve/main/google_gemma-4-31B-it-Q4_K_M.gguf'"
  exit 1
fi

nohup numactl --interleave=all "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias gemma-4-31B-it \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 16384 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started gemma-4-31B-it PID=$! port=${PORT} binary=${LLAMA_SERVER}"
