#!/bin/bash
# Start Qwen3.6-35B-A3B MoE on vasp-01:8081 (GPU).
#
# 35B total / 3B active per token. All weights GPU-resident (A3B experts
# are tiny — CPU-offload would only slow this down on a 32GB V100S).
# Q4_K_M ~20.6GB weights + q8_0 KV at 64K ctx ~6-7GB = ~27GB, fits V100S 32GB.
#
# Config is tuned from upstream guidance (Unsloth / llama.cpp upstream / Qwen
# model card) rather than copy-paste defaults. See the comments below for
# each non-obvious flag.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-mmq}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference.log}"
QUANT="${QUANT:-UD-Q4_K_M}"
MODEL_FILE="/scratch/ai/models/Qwen3.6-35B-A3B-${QUANT}.gguf"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server binary not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port 8081( |$)/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 3
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "To fetch: huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \\" >&2
  echo "            'Qwen3.6-35B-A3B-${QUANT}.gguf' \\" >&2
  echo "            --local-dir /scratch/ai/models/" >&2
  exit 1
fi

# Flag notes:
#   --ctx-size 65536      — 64K of Qwen3.6's native 262K context. No YaRN needed.
#   --n-gpu-layers 999    — all weights on GPU. Per Doctor-Shotgun MoE offload
#                           guide, -ot "ffn_.*_exps=CPU" is only for models
#                           that don't fit. A3B fits with room to spare.
#   --batch-size 4096     — Unsloth Qwen3 bench shows +43% pp vs 2048
#     --ubatch-size 4096
#   --cache-type-k q8_0   — q4_0 KV is flaky with -fa (ik_llama #1142,
#   --cache-type-v q8_0     llama.cpp #19036). q8_0 is the V100-safe sweet
#                           spot; ~2× VRAM of q4_0 but still fits at 64K.
#   -fa on                — V100 uses sm_70 MMA kernels (PR #13194). Safe
#                           when built with CMAKE_CUDA_ARCHITECTURES=70 AND
#                           KV cache is NOT q4_0/q5_0.
#   --parallel 1          — llama.cpp DIVIDES ctx among slots (see #17989).
#                           --parallel 2 at 65536 ctx = 2×32K slots, NOT two
#                           clients sharing 64K. Use 1 for single-stream.
#   --reasoning-format none          — correct flag name (not --reasoning off).
#   --chat-template-kwargs enable_thinking=false — Qwen-native switch to skip
#                           thinking generation entirely (PR #11607).
nohup numactl --interleave=all "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Qwen3.6-35B-A3B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 999 \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt -fa on --parallel 1 --mlock --cont-batching --metrics --jinja \
  --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking":false}' \
  > "${LOG_PATH}" 2>&1 &

echo "Started Qwen3.6-35B-A3B (${QUANT}) PID=$! binary=${LLAMA_SERVER}"
echo "Client-side sampling (coding, thinking off): temperature=0.7, top_p=0.8,"
echo "  top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
