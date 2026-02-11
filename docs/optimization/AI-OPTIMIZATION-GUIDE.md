# Advanced AI Optimization Guide - 3-Node HPC Cluster

**Date:** October 27, 2025
**Hardware:** 3x Dell servers with V100S GPUs + 100GbE InfiniBand
**Focus:** Maximum performance for AI inference, training, and HPC workloads

---

## Executive Summary

This cluster is capable of **enterprise-grade distributed AI workloads** with proper configuration:

- **GPUDirect RDMA** - 2-10x faster GPU-to-GPU communication
- **vLLM Distributed Serving** - Serve models >32GB across 3 GPUs with tensor parallelism
- **SGLang HiCache** - Advanced KV caching for transformer models
- **PyTorch + TensorRT** - Maximum inference performance on V100S
- **Ray/Kubernetes** - Production-grade orchestration

**Performance potential:**
- Distributed inference: ~12 GB/s inter-GPU bandwidth
- Training: Sub-microsecond GPU communication latency
- HPC: Full InfiniBand RDMA for MPI workloads

---

## 1. GPUDirect RDMA Configuration

### What is GPUDirect RDMA?

GPUDirect RDMA allows GPUs to **directly access the InfiniBand NIC's memory**, bypassing the CPU and system RAM entirely. This is critical for multi-node GPU workloads.

### Performance Gains

- **Latency:** 2-10x reduction compared to CPU-mediated transfers
- **Bandwidth:** Near line-rate (approaching 100 Gbps)
- **CPU Overhead:** Zero - GPU talks directly to network

### Hardware Support

✅ **Tesla V100S GPUs** - Full GPUDirect RDMA support
✅ **Mellanox SB7800 Switch** - InfiniBand EDR with RDMA
✅ **Dell Servers** - IOMMU/VT-d enabled in BIOS

### Configuration Steps

#### 1. Install Mellanox OFED

```bash
# Already included in configure-infiniband.sh
apt install -y infiniband-diags ibverbs-utils rdma-core perftest
```

#### 2. Install CUDA 12.x with GPUDirect Support

```bash
# Download CUDA 12.x from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# Verify CUDA version
nvcc --version
```

#### 3. Enable IOMMU in BIOS

```bash
# Already documented in BIOS settings
# Verify IOMMU is enabled:
dmesg | grep -i iommu
```

#### 4. Verify GPUDirect RDMA

```bash
# Check GPU topology
nvidia-smi topo -m

# Should show "PIX" for GPU-to-GPU over PCIe
# Should show InfiniBand devices connected

# Test bandwidth and latency
cd /usr/local/cuda/samples/p2pBandwidthLatencyTest
make
./p2pBandwidthLatencyTest
```

#### 5. Configure NCCL for InfiniBand

```bash
# Add to /etc/environment or ~/.bashrc
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
```

---

## 2. vLLM Distributed Serving

### Overview

vLLM is a high-throughput inference server that supports **tensor parallelism across multiple nodes** via Ray cluster.

### Use Cases

- Serve models larger than 32GB (e.g., LLaMA-70B, Mixtral-8x7B)
- Distribute inference load across 3 GPUs
- Low-latency API serving with InfiniBand backend

### Architecture

```
┌─────────────────────────────────────────────┐
│           vLLM API Server                   │
│         (receives HTTP requests)            │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │      Ray Cluster          │
    │   (tensor parallelism)    │
    └──┬──────────┬──────────┬──┘
       │          │          │
   ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
   │ GPU 1 │  │ GPU 2 │  │ GPU 3 │
   │ V100S │  │ V100S │  │ V100S │
   │ 32GB  │  │ 32GB  │  │ 32GB  │
   └───────┘  └───────┘  └───────┘
       │          │          │
   ┌───┴──────────┴──────────┴───┐
   │  InfiniBand (100GbE RDMA)   │
   │    Sub-microsecond latency  │
   └─────────────────────────────┘
```

### Installation

```bash
# Install vLLM on all 3 nodes
pip install vllm

# Install Ray for distributed serving
pip install ray[default]
```

### Ray Cluster Setup

**Server 1 (pve1) - Ray Head:**
```bash
# Start Ray head node with InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

ray start --head \
  --node-ip-address=10.100.0.1 \
  --port=6379 \
  --num-gpus=1
```

**Server 2 (pve2) - Ray Worker:**
```bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

ray start --address='10.100.0.1:6379' \
  --node-ip-address=10.100.0.2 \
  --num-gpus=1
```

**Server 3 (pve3) - Ray Worker:**
```bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

ray start --address='10.100.0.1:6379' \
  --node-ip-address=10.100.0.3 \
  --num-gpus=1
```

### Launch vLLM with Tensor Parallelism

**On Server 1 (Ray head):**
```bash
vllm serve meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 3 \
  --host 10.100.0.1 \
  --port 8000
```

This splits the 70B model across 3x V100S GPUs!

### Test Distributed Inference

```bash
curl http://10.100.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-70b-chat-hf",
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 100
  }'
```

---

## 3. SGLang with HiCache

### What is HiCache?

**HiCache** is SGLang's two-level caching system for transformer Key-Value (KV) pairs:

1. **GPU Cache** - Small, ultra-fast cache on GPU for hot data
2. **CPU Cache** - Larger cache in system RAM for warm data

### Benefits

- **Reduced Recomputation** - Cached KV pairs don't need recalculation
- **Higher Throughput** - Fewer GPU cycles wasted on redundant work
- **Better Memory Usage** - Intelligently evicts cold data to CPU

### How InfiniBand Helps

While HiCache doesn't directly use InfiniBand for caching, the fast interconnect is **critical for distributed cache coordination**:
- Synchronizing cache state across nodes
- Efficient cache invalidation
- Load balancing cached requests

### Installation

```bash
pip install "sglang[all]"
```

### Usage

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-2-13b-chat-hf \
  --host 10.100.0.1 \
  --port 30000 \
  --mem-fraction-static 0.8
```

---

## 4. PyTorch + TensorRT (Maximum Performance)

### Overview

Combine **PyTorch's ease of use** with **TensorRT's raw performance** using `torch.compile`.

### What You Get

- **Graph Optimizations** - Kernel fusion, operation reordering
- **Quantization** - INT8/FP16 for faster inference
- **TensorRT Backend** - NVIDIA's optimized inference engine

### Installation

```bash
# Install PyTorch 2.0+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TensorRT
pip install tensorrt

# Install torch-tensorrt
pip install torch-tensorrt
```

### Usage Example

```python
import torch
import torch_tensorrt

# Load your PyTorch model
model = torch.load("model.pth").eval().cuda()

# Compile with TensorRT backend
trt_model = torch.compile(
    model,
    backend="torch_tensorrt",
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float16},  # Use FP16 for V100S
    }
)

# Inference is now TensorRT-optimized
with torch.no_grad():
    output = trt_model(input_tensor)
```

---

## 5. Other High-Performance Frameworks

### Distributed Inference

**NVIDIA Triton Inference Server**
```bash
# Multi-framework support (PyTorch, TensorFlow, ONNX)
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  nvcr.io/nvidia/tritonserver:24.10-py3
```

**DeepSpeed-Inference**
```bash
pip install deepspeed

# Launch with tensor parallelism
deepspeed --num_gpus 3 inference.py --model llama-70b
```

### Distributed Training

**DeepSpeed (ZeRO Optimizer)**
```python
import deepspeed

# DeepSpeed config with ZeRO stage 3
ds_config = {
    "train_batch_size": 64,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

**Megatron-LM (NVIDIA)**
```bash
# Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git

# Train with model + data parallelism
python pretrain_gpt.py \
  --tensor-model-parallel-size 3 \
  --pipeline-model-parallel-size 1
```

**Horovod (Multi-Framework)**
```bash
# Install Horovod with NCCL support
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

# Run distributed training
horovodrun -np 3 -H 10.100.0.1:1,10.100.0.2:1,10.100.0.3:1 \
  python train.py
```

### HPC and Scientific Computing

**MPI (Open MPI / MVAPICH2)**
```bash
# Install Open MPI with InfiniBand support
apt install openmpi-bin libopenmpi-dev

# Run MPI program across 3 nodes
mpirun -np 3 \
  -host 10.100.0.1,10.100.0.2,10.100.0.3 \
  --mca btl openib,self,vader \
  ./my_program
```

**JAX (High-Performance ML)**
```python
import jax
import jax.numpy as jnp

# JAX automatically uses all available GPUs
devices = jax.devices()  # Shows 3 GPUs

# Distributed computation with pmap
@jax.pmap
def parallel_function(x):
    return x ** 2

result = parallel_function(jnp.arange(3))
```

---

## 6. Container Orchestration

### Option 1: Ray (AI-Focused)

**Best for:** Python-heavy AI workloads, distributed inference

```bash
# Install Ray on all nodes
pip install ray[default]

# Deploy vLLM, SGLang, custom models via Ray
ray start --head --node-ip-address=10.100.0.1
```

### Option 2: Kubernetes + NVIDIA GPU Operator

**Best for:** Multi-tenant, production-grade deployments

```bash
# Install Kubernetes
curl -sfL https://get.k3s.io | sh -

# Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml
```

### Option 3: Slurm (Traditional HPC)

**Best for:** Batch scheduling, traditional HPC workloads

```bash
# Install Slurm
apt install slurm-wlm

# Submit job to all 3 nodes
sbatch --nodes=3 --ntasks-per-node=1 --gres=gpu:1 job.sh
```

---

## 7. Complete Software Stack

### Recommended Configuration

```
┌─────────────────────────────────────────────┐
│          Application Layer                  │
│  vLLM | SGLang | Triton | Custom Models    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│         Framework Layer                     │
│  PyTorch 2.0+ | JAX | TensorFlow           │
│  + torch.compile + TensorRT backend        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│      Acceleration Layer                     │
│  NCCL | cuDNN | TensorRT | DeepSpeed       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│         CUDA Layer                          │
│  CUDA 12.x + GPUDirect RDMA                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────┴───────────────────────────┐
│       Hardware Layer                        │
│  3x V100S (32GB) + InfiniBand EDR          │
└─────────────────────────────────────────────┘
```

### Installation Script

```bash
#!/bin/bash
# Complete AI stack installation

# CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# PyTorch 2.0+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorRT
pip install tensorrt torch-tensorrt

# Distributed frameworks
pip install vllm ray[default] deepspeed sglang[all]

# HPC tools
apt install openmpi-bin libopenmpi-dev

# NCCL environment
echo 'export NCCL_SOCKET_IFNAME=ib0' >> ~/.bashrc
echo 'export NCCL_IB_DISABLE=0' >> ~/.bashrc
echo 'export NCCL_IB_GID_INDEX=3' >> ~/.bashrc
```

---

## 8. Performance Benchmarking

### GPU-to-GPU Bandwidth (InfiniBand RDMA)

```bash
# NCCL bandwidth test
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### vLLM Throughput

```bash
# Benchmark inference throughput
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 3

# Run benchmark
python benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Llama-2-13b-chat-hf \
  --num-prompts 1000
```

### MPI Communication Latency

```bash
# OSU MPI benchmarks
wget http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
tar xf osu-micro-benchmarks-7.4.tar.gz
cd osu-micro-benchmarks-7.4
./configure
make

# Run latency test
mpirun -np 2 -host 10.100.0.1,10.100.0.2 ./pt2pt/osu_latency
```

---

## 9. Next Steps

### Immediate (Essential)

1. **Enable GPUDirect RDMA** - Install CUDA 12.x, verify with `nvidia-smi topo -m`
2. **Configure NCCL** - Set environment variables for InfiniBand
3. **Deploy Ray Cluster** - Set up distributed serving infrastructure

### Short-term (High Value)

4. **Deploy vLLM** - Start serving models with tensor parallelism
5. **Install PyTorch + TensorRT** - Maximum inference performance
6. **Benchmark Performance** - Measure actual throughput and latency

### Long-term (Advanced)

7. **Production Orchestration** - Kubernetes or Slurm for multi-tenant workloads
8. **Custom Models** - Deploy your own fine-tuned models
9. **Advanced Training** - DeepSpeed or Megatron-LM for large-scale training

---

## Summary

Your 3-node cluster is capable of **enterprise-grade AI workloads** with the right configuration:

✅ **GPUDirect RDMA** - 2-10x faster GPU communication
✅ **vLLM** - Serve 70B models across 3 GPUs
✅ **TensorRT** - Maximum inference performance
✅ **Ray/K8s** - Production-grade orchestration
✅ **InfiniBand** - Sub-microsecond latency

**This is a serious HPC cluster** - not just a hobbyist setup!
