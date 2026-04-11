#!/bin/bash
# Start Qwen3.5-27B dense on vasp-01.
# Dense model: all weights on GPU, no expert offload needed.
# Q4_K_M ~16.5GB fits in V100S 32GB with ~15GB for KV cache.
# --parallel 2: 32K context per slot (ctx-size/parallel). parallel=4 gave only 8K/slot.
# Uses native binary (b8692+) for Gemma 4 compat and latest optimizations.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-gemma4}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"
QUANT="${QUANT:-Q4_K_M}"
MODEL_FILE="/scratch/ai/models/Qwen3.5-27B-${QUANT}.gguf"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server binary not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port 8081( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  exit 1
fi

nohup numactl --interleave=all "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Qwen3.5-27B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen3.5-27B (${QUANT}) PID=$! binary=${LLAMA_SERVER}"
