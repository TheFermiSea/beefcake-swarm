#!/bin/bash
# Start nomic-embed-text-v1.5 embedding service on CPU (port 8082).
# Runs alongside the main LLM inference on port 8081.
# Provides OpenAI-compatible /v1/embeddings endpoint.
# Model: 139MB Q8_0, runs entirely on CPU — zero GPU impact.
set -euo pipefail

CONTAINER="${CONTAINER:-/cluster/shared/containers/llama-server.sif}"
LOG_PATH="${LOG_PATH:-/tmp/llama-embedding.log}"
MODEL_FILE="/scratch/ai/models/nomic-embed-text-v1.5.Q8_0.gguf"
PORT="${EMBEDDING_PORT:-8082}"

command -v apptainer >/dev/null
mkdir -p /tmp/cuda-cache

export APPTAINERENV_HOME=/tmp
export APPTAINERENV_CUDA_CACHE_PATH=/tmp/cuda-cache
# Hide GPU to prevent CUDA memory probe crash when GPU is full from main model.
export CUDA_VISIBLE_DEVICES=

# Kill only embedding processes (port 8082), not the main LLM server (8081)
existing_pids="$(ps -eo pid=,args= | awk '/llama-server.*--port '"$PORT"'/ { print $1 }')"
if [[ -n "${existing_pids}" ]]; then
  kill ${existing_pids} 2>/dev/null || true
  sleep 2
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "ERROR: Embedding model not found: ${MODEL_FILE}" >&2
  exit 1
fi

# Run on CPU only (--n-gpu-layers 0) to avoid competing with the main model.
# --embedding enables the /v1/embeddings endpoint.
# --parallel 8: embedding requests are cheap, serve many concurrently.
# --ctx-size 8192: nomic-embed-text-v1.5 supports up to 8192 tokens.
nohup numactl --membind=1 apptainer run --nv --bind /scratch/ai:/scratch/ai:ro "${CONTAINER}" \
  --model "${MODEL_FILE}" \
  --alias nomic-embed-text-v1.5 \
  --host 0.0.0.0 --port "${PORT}" \
  --embedding \
  --n-gpu-layers 0 -fit off \
  --ctx-size 8192 --batch-size 8192 \
  --threads 8 \
  --parallel 8 \
  > "${LOG_PATH}" 2>&1 &

echo "Started nomic-embed-text-v1.5 (CPU embedding) PID=$! port=${PORT}"
