#!/bin/bash
# Start Gemma-4-31B-it on vasp-02:8081 (GPU).
# Dense 30.7B, Q4_K_M ~18.3GB. Fits V100S 32GB with ~10GB KV headroom.
# Native 256K context; capped here at 32K due to q8_0 KV VRAM cost.
#
# Config tuned from Google's Gemma-4 model card + Unsloth Gemma-4 guide +
# llama.cpp upstream issue tracker. Each non-default flag has a rationale
# in the comments below.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-latest-20260418}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-gemma4-31b.log}"
MODEL_FILE="/scratch/ai/models/google_gemma-4-31B-it-Q4_K_M.gguf"
PORT="${GEMMA_PORT:-8081}"

# CUDA runtime (nvhpc_sdk 24.11 ships CUDA 12.6). The new llama-server binary
# was built without RPATH, so we resolve libcudart/libcublas via env.
CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11"
export LD_LIBRARY_PATH="${CUDA_ROOT}/cuda/12.6/lib64:${CUDA_ROOT}/cuda/12.6/compat:${CUDA_ROOT}/math_libs/12.6/targets/x86_64-linux/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: Gemma4-capable llama-server not found: ${LLAMA_SERVER}" >&2
  echo "Build on the target node with gcc-toolset-13: see" >&2
  echo "  dnf install -y gcc-toolset-13" >&2
  echo "  source /opt/rh/gcc-toolset-13/enable" >&2
  echo "  cd /root/llama.cpp && cmake -B build-latest -DGGML_CUDA=ON ..." >&2
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
  echo "Download: huggingface-cli download bartowski/google_gemma-4-31B-it-GGUF \\" >&2
  echo "          'google_gemma-4-31B-it-Q4_K_M.gguf' --local-dir /scratch/ai/models/" >&2
  exit 1
fi

# Flag notes:
#   --ctx-size 32768      — q8_0 KV at 32K ≈ 3GB. Bump to 65536 if workload
#                           demands it and you're willing to spend the VRAM.
#   --cache-type-k q8_0   — q4_0 KV is flaky with -fa (ik_llama #1142). q8_0
#   --cache-type-v q8_0     is the V100-safe sweet spot.
#   -fa on                — V100 sm_70 MMA kernels; safe with q8_0 KV.
#   (no --swa-full)       — Gemma-4's interleaved attention: local layers use
#                           a 1024-token sliding window, global layers see the
#                           full ctx. Passing --swa-full would force all layers
#                           to materialize the full cache, costing ~14 GB extra
#                           VRAM at 32K ctx — pushes Q4_K_M + q8_0 KV past the
#                           V100S 32GB budget (confirmed OOM on first attempt).
#                           Leaving SWA default: model still reasons over full
#                           context via global layers; local layers do local
#                           attention, which is fine for agentic coding.
#                           Also avoids bug #21468 (-fa + --swa-full + cache
#                           reuse interaction).
#   --reasoning off       — suppresses Gemma-4's <|channel>thought…<channel|>
#                           reasoning output; without it, chat response
#                           `content` is empty. Do NOT add --reasoning-format
#                           none or --chat-template-kwargs enable_thinking=false:
#                           the first leaks the raw channel tokens into
#                           content, the second is a Qwen-specific jinja kwarg
#                           that Gemma's template doesn't recognize. Tested
#                           empirically — `--reasoning off` alone gives clean
#                           text output.
#   --parallel 1          — single-stream for now; bump only if you run a
#                           batched workload.
nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias Gemma-4-31B-it \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 999 \
  --threads 16 --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on --parallel 1 \
  --mlock --cont-batching --metrics --jinja \
  --reasoning off \
  > "${LOG_PATH}" 2>&1 &

echo "Started Gemma-4-31B-it PID=$! port=${PORT} binary=${LLAMA_SERVER}"
echo "Expected warm-up: ~15-20s"
echo "Client-side sampling (Google card): temperature=1.0, top_p=0.95, top_k=64"
echo "  (temperature 1.0 is load-bearing for Gemma; lower values degrade output)"
