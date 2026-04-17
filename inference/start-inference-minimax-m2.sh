#!/bin/bash
# Start MiniMax-M2.7 as the Strategist "slow-smart oracle" tier.
#
# Profile: ~300B total params, ~9B active per token, 192K context.
# Architecture: MoE with 256 experts (8 active per token).
#
# Deployment strategy — MoE CPU-offload:
#   - Attention + norms + embeddings on GPU (~13-17 GB at Q3)
#   - All 256 expert FFNs on CPU (~110-150 GB system RAM)
#   - Throughput: ~6 tok/s output (DDR4 bandwidth-bound). Expected latency
#     per Strategist call: 3-8 min. Only invoked for final escalation.
#
# Quant: Q3_K_M (~130 GB). Fits 256 GB host RAM with headroom. Lower quant
# (Q2_K_L ~95 GB) is faster to load but degrades reasoning quality.
#
# Model_type=minimax_m2 is a NEW architecture. Verify llama.cpp support
# before deploying — upstream merge happened ~2026-03, check binary version.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server-mmq}"
LOG_PATH="${LOG_PATH:-/tmp/llama-inference-minimax.log}"
QUANT="${QUANT:-Q3_K_M}"
MODEL_FILE="${MODEL_FILE:-/scratch/ai/models/MiniMax-M2.7-${QUANT}.gguf}"
PORT="${PORT:-8084}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server binary not found: ${LLAMA_SERVER}" >&2
  exit 1
fi

# Precheck: verify the running llama.cpp supports minimax_m2 arch
if ! "${LLAMA_SERVER}" --help 2>&1 | grep -q 'minimax\|mimo' && \
   ! "${LLAMA_SERVER}" --version 2>&1 | grep -qE 'b(8[3-9]|9)[0-9]{2}'; then
  echo "WARN: llama-server may not support minimax_m2 architecture." >&2
  echo "      Verify with: ${LLAMA_SERVER} --list-models 2>&1 | grep -i minimax" >&2
fi

# Stop any prior instance on this port
existing_pids="$(ps -eo pid=,args= | awk -v port="--port ${PORT}" '$0 ~ "llama-server.*" port "( |$)" { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 3  # MiniMax takes longer to release CPU-mapped expert memory
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Model file not found: ${MODEL_FILE}" >&2
  echo "To fetch (Q3_K_M ~130 GB):" >&2
  echo "  huggingface-cli download MiniMaxAI/MiniMax-M2.7 --include '*${QUANT}*.gguf' \\" >&2
  echo "                          --local-dir /scratch/ai/models/" >&2
  exit 1
fi

# Key flags for MoE CPU-offload:
#   --n-gpu-layers 999      : all transformer layers on GPU (attention/norms)
#   --override-tensor       : regex match puts expert FFN tensors on CPU
#   --threads 48            : CPU threads for expert FFN compute
#   --no-mmap               : keep experts resident (no page faults under load)
#   --ctx-size 32768        : 32K is generous for Strategist prompts; trim if KV
#                             cache is tight alongside attention weights on GPU
nohup numactl --interleave=all "${LLAMA_SERVER}" \
  --model "${MODEL_FILE}" \
  --alias MiniMax-M2.7 \
  --host 0.0.0.0 --port "${PORT}" \
  --ctx-size 32768 --n-gpu-layers 999 \
  --override-tensor '\.ffn_(gate|up|down)_exps\.weight=CPU' \
  --threads 48 --batch-size 512 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --mlock --no-mmap --cont-batching --metrics --jinja \
  > "${LOG_PATH}" 2>&1 &

echo "Started MiniMax-M2.7 (${QUANT}) PID=$! port=${PORT} binary=${LLAMA_SERVER}"
echo "Expected warm-up: ~2-3 min to load ${MODEL_FILE}"
echo "Expected throughput: ~6 tok/s output (CPU-bound MoE routing)"
