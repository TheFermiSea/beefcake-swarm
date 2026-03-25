#!/bin/bash
# Start Qwen3.5-122B-A10B on vasp-01 (MoE coder, expert-offload)
set -e
export LD_LIBRARY_PATH=/usr/local/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/REDIST/cuda/12.6/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/REDIST/math_libs/12.6/targets/x86_64-linux/lib:/opt/rh/gcc-toolset-13/root/usr/lib64
export HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache GGML_CUDA_DISABLE_FUSION=1

pkill -9 llama-server-mmq 2>/dev/null || true
sleep 2

nohup numactl --interleave=all /usr/local/bin/llama-server-mmq \
  --model /scratch/ai/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  --alias Qwen3.5-122B-A10B \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 --n-gpu-layers 99 \
  -ot ".ffn_.*_exps.=CPU" \
  --threads 32 --batch-size 512 --ubatch-size 512 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-prompt -fa on --parallel 2 --mlock --cont-batching --metrics --jinja \
  > /tmp/llama-inference.log 2>&1 &

echo "Started Qwen3.5-122B-A10B PID=$!"
