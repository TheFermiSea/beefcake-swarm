#!/bin/bash
# Start Qwen3-Coder-Next on vasp-03 (80B/3B MoE, expert-offload)
set -e
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-unified/lib64:/opt/rh/gcc-toolset-13/root/usr/lib64
export HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache GGML_CUDA_DISABLE_FUSION=1

pkill -9 llama-server-mmq 2>/dev/null || true
sleep 2

# Start server with Flash Attention (native support on V100S/Volta via llama.cpp GGML kernels)
nohup numactl --interleave=all /usr/local/bin/llama-server-mmq \
  --model /scratch/ai/models/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --alias Qwen3-Coder-Next \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 99 \
  -ot ".ffn_.*_exps.=CPU" \
  --threads 32 --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 4 --mlock --cont-batching --metrics --jinja \
  > /tmp/llama-inference.log 2>&1 &

echo "Started Qwen3-Coder-Next PID=$!"
