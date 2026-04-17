#!/bin/bash
# Start Qwen3.6-35B-A3B MoE on vasp-01 (replaces Qwen3.5-27B dense).
#
# 35B total / 3B active per token. All weights GPU-resident; the A3B routing
# makes inference ~2× faster than dense Qwen3.5-27B (target: ~45-55 tok/s).
# model_type=qwen3_5_moe is already supported by llama.cpp (same family as
# the existing Qwen3.5 build — no new binary needed).
#
# Q4_K_M ~20GB fits in V100S 32GB with ~10-12GB for KV cache at 64K context.
# Drop to Q3_K_M (~16GB) if KV cache needs more headroom.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-mmq}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"
QUANT="${QUANT:-UD-Q4_K_M}"
MODEL_FILE="/scratch/ai/models/Qwen3.6-35B-A3B-${QUANT}.gguf"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server binary not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

# Kill any existing llama-server on port 8081 before restarting.
existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port 8081( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "To fetch: huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \\" >&2
  echo "            'Qwen3.6-35B-A3B-${QUANT}.gguf' \\" >&2
  echo "            --local-dir /scratch/ai/models/" >&2
  exit 1
fi

nohup numactl --interleave=all "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen3.6-35B-A3B (${QUANT}) PID=$! binary=${LLAMA_SERVER}"
