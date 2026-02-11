# Distributed vLLM/llama.cpp Cluster - Session Progress

**Date**: January 16, 2026
**Project**: Distributed inference cluster for OR1-Behemoth 72B model using llama.cpp with RPC

## Summary

Successfully deployed llama.cpp across 3 V100S nodes with RPC support for distributed inference. RPC servers are running on nodes 2 & 3, and OR1-Behemoth Q4_K_M model has been downloaded (47.4 GB). Ready to launch distributed inference server.

## Completed Tasks

### 1. llama.cpp Build on Node1 (vllm-node1)
- **Status**: ✅ Complete
- **Location**: `/opt/llama.cpp`
- **Build**: GCC 12.2.1, CUDA 12.2, GGML_CUDA=ON, GGML_RPC=ON
- **Binaries**:
  - `/opt/llama.cpp/build/bin/llama-server` (7 MB)
  - `/opt/llama.cpp/build/bin/rpc-server` (184 KB)
  - `/opt/llama.cpp/build/bin/llama-cli` (5.3 MB)
- **Version**: 7760 (commit 388ce8224)
- **Hardware**: Tesla V100S-PCIE-32GB (Compute Capability 7.0)

**Build Issues Resolved**:
- Initial build failed with GCC 8.5 - incomplete `std::filesystem` support
- Installed `gcc-toolset-12` (GCC 12.2.1)
- Performed clean rebuild from scratch with explicit compiler paths

### 2. llama.cpp Deployment to Nodes 2 & 3
- **Status**: ✅ Complete
- **Method**: Prebuilt binaries copied from node1 (avoided rebuild)
- **Location**: `/opt/llama.cpp/build/bin/` on both nodes
- **Verification**: Both RPC servers detect V100S GPU and run successfully

**Deployment Process**:
```bash
# On node1:
cd /opt && tar czf /tmp/llama-cpp-build.tar.gz llama.cpp/build/bin

# Copy via pve hosts:
# pve2 → node2 (10.0.0.32)
# pve3 → node3 (10.0.0.33)

# On nodes 2 & 3:
mkdir -p /opt
cd /opt && tar xzf /tmp/llama-cpp-build.tar.gz
```

### 3. RPC Servers (Nodes 2 & 3)
- **Status**: ✅ Running
- **Node2** (10.0.0.32): PID 25366, listening on 0.0.0.0:50052
- **Node3** (10.0.0.33): PID 40349, listening on 0.0.0.0:50052
- **GPU**: Tesla V100S-PCIE-32GB on each node

**Command Used**:
```bash
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
nohup /opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc.log 2>&1 &
```

**Verification**:
```bash
ss -tlnp | grep 50052
ps aux | grep rpc-server
```

## In Progress

### 4. Model Download (Node1)
- **Status**: ✅ Complete
- **File**: `/opt/models/OR1-Behemoth.Q4_K_M.gguf`
- **Size**: 47.4 GB (47,369,119,648 bytes)
- **Download Time**: 16m48s (average 44.8 MB/s)
- **Source**: https://huggingface.co/mradermacher/OR1-Behemoth-GGUF
- **Completed**: 19:03 (January 16, 2026)

**Command**:
```bash
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q4_K_M.gguf" -O OR1-Behemoth.Q4_K_M.gguf
```

**Disk Space**: 99 GB available on node1 (sufficient)

## Pending Tasks

### 5. Launch llama-server on Node1
- **Command**:
```bash
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
cd /opt/llama.cpp/build/bin

./llama-server \
    -m /opt/models/OR1-Behemoth.Q4_K_M.gguf \
    --rpc 10.100.0.32:50052,10.100.0.33:50052 \
    -ngl 999 \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 4096 \
    -ngl 999 \
    --parallel 3
```

**Important**: Use **InfiniBand IPs** for RPC backends:
- Node2: `10.100.0.32:50052`
- Node3: `10.100.0.33:50052`

### 6. Test Inference
- **Endpoint**: `http://10.0.0.31:8000/v1/chat/completions`
- **Test Command**:
```bash
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Write a Rust function to calculate fibonacci numbers"}],
    "max_tokens": 256
  }'
```

## Infrastructure Details

### Node Access
```bash
# Proxmox Hosts (Tailscale)
ssh root@100.127.208.104  # pve1 → vllm-node1
ssh root@100.127.30.114   # pve2 → vllm-node2
ssh root@100.68.22.98     # pve3 → vllm-node3

# VMs (via pve hosts)
ssh root@10.0.0.31  # vllm-node1 (head node)
ssh root@10.0.0.32  # vllm-node2 (RPC worker)
ssh root@10.0.0.33  # vllm-node3 (RPC worker)
```

### Network Configuration
| Node | Eth IP | IB IP | Role |
|------|--------|-------|------|
| vllm-node1 | 10.0.0.31 | 10.100.0.31 | Head server |
| vllm-node2 | 10.0.0.32 | 10.100.0.32 | RPC worker |
| vllm-node3 | 10.0.0.33 | 10.100.0.33 | RPC worker |

**Important**: Use InfiniBand IPs (10.100.0.x) for RPC communication - 100Gb/s vs 1Gb/s Ethernet.

### Hardware
- **CPU**: 40 cores Xeon Gold 6248 (HT disabled)
- **GPU**: Tesla V100S 32GB (Compute Capability 7.0)
- **RAM**: ~376 GB per node
- **IB**: ConnectX-6 HDR100 @ 100Gb/s

### Software Stack
- **OS**: Rocky Linux 8.10
- **NVIDIA Driver**: 535.183.01 (CUDA 12.2)
- **CUDA**: `/usr/local/cuda-12.2`
- **llama.cpp**: v7760 built with GCC 12.2.1
- **GCC**: gcc-toolset-12 (GCC 12.2.1)

## Model Information
- **Name**: OR1-Behemoth
- **Parameters**: 72B (Qwen3ForCausalLM)
- **Quantization**: Q4_K_M
- **File Size**: 47.4 GB
- **HuggingFace**: https://huggingface.co/mradermacher/OR1-Behemoth-GGUF
- **Use Case**: Rust/programming reasoning model

## Verification Commands

```bash
# Check RPC servers
for node in 32 33; do
  echo "=== 10.0.0.$node ==="
  ssh -J root@100.127.208.104 root@10.0.0.$node "ps aux | grep rpc-server | grep -v grep; ss -tlnp | grep 50052"
done

# Check download progress
ssh -J root@100.127.208.104 root@10.0.0.31 "ls -lh /opt/models/*.gguf; cat /tmp/download.log | tail -10"

# Check GPU status
for node in 31 32 33; do
  echo "=== 10.0.0.$node ==="
  ssh -J root@100.127.208.104 root@10.0.0.$node "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader"
done

# Verify binaries
for node in 31 32 33; do
  echo "=== 10.0.0.$node ==="
  ssh -J root@100.127.208.104 root@10.0.0.$node "ls -lh /opt/llama.cpp/build/bin/{llama-server,rpc-server}"
done
```

## Known Constraints

### 1. V100 GPU Limitations

**Critical Issue**: Tesla V100S (Volta, CC 7.0) lacks INT4 tensor cores, making Q4 quantization suboptimal.

| GPU Architecture | INT4 Tensor Cores | Q4 Performance |
|------------------|-------------------|----------------|
| **V100** (CC 7.0) | ❌ No | ⚠️ Slow (de-quantization on CUDA cores) |
| A100 (Ampere) | ✅ Yes | ✅ Fast (native INT4 ops) |
| H100 (Hopper) | ✅ Yes | ✅ Fast (native INT4 ops) |

**Impact on Q4 Models**:
- Q4 quantized weights must be de-quantized INT4→FP16 before matrix multiplication
- V100 performs this on CUDA cores (not tensor cores), adding significant overhead
- Same Q4 model runs **3-4× faster on A100** vs V100

**Why We're Still Using Q4**:
```
FP16 Model: 72B × 2 bytes = 144 GB VRAM (impossible)
Q4 Model: 72B × 0.5 bytes = 36 GB VRAM (fits in 3×32GB)
```

FP16 is impossible with current hardware. Q4 is the **only viable option** for 72B model.

### 2. Memory Constraints

| Quantization | Model Size | Per GPU (3×V100S) | V100 Performance |
|--------------|------------|-------------------|------------------|
| **Q4_K_M** | 47.4 GB | 16 GB | ⚠️ Slow (no INT4 cores) |
| **Q4_K_S** | 43.8 GB | 15 GB | 🟡 Slightly better than Q4_K_M |
| **Q5_K_S** | 51.4 GB | 17 GB | ✅ Better (less de-quantization) |
| **Q6_K** | 64.3 GB | 21 GB | ✅ Good balance |
| **Q8_0** | 77.3 GB | 26 GB | ✅ Best (close to FP16 performance) |

**Q5_K_S or Q6_K would be better for V100** but still requires testing for KV cache headroom.

### 3. Network Constraints
- InfiniBand must be used for RPC communication
- IB IPs: 10.100.0.x
- MTU: 2044 (IPoIB) / 4096 (Hardware)

## Alternative Quantizations Available

The OR1-Behemoth model has multiple quantization options available on HuggingFace:

### Recommended for V100S (Better Performance)

**Q5_K_S** (51.4 GB, split in 2 parts):
- Higher precision = less de-quantization overhead
- Still fits with room for KV cache
- Better quality than Q4
- Download: `OR1-Behemoth.Q5_K_S.gguf.part1of2` + `OR1-Behemoth.Q5_K_S.gguf.part2of2`

**Q6_K** (64.3 GB, split in 2 parts):
- Marked as "very good quality" by author
- Significantly less de-quantization overhead than Q4
- Fits on 3×32GB (21 GB/GPU)
- Recommended if V100 performance is critical

**Q8_0** (77.3 GB, split in 2 parts):
- Marked as "fast, best quality" by author
- Close to FP16 performance
- Fits but tight (25.8 GB/GPU, ~6 GB for KV cache)
- Best option if memory allows

### Current Download (Q4_K_M)
- Status: In progress (46% complete)
- Size: 47.4 GB
- Fit: 16 GB/GPU (comfortable)
- Issue: Suboptimal on V100 due to INT4 de-quantization

### Switching to Different Quantization

If Q4 performance is unacceptable, switch to a higher precision model:

```bash
# Stop current download
ssh root@10.0.0.31 "pkill wget && rm /opt/models/OR1-Behemoth.Q4_K_M.gguf"

# Download Q5_K_S (better quality, better V100 performance)
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q5_K_S.gguf.part1of2"
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q5_K_S.gguf.part2of2"

# Concatenate parts
cat OR1-Behemoth.Q5_K_S.gguf.part1of2 OR1-Behemoth.Q5_K_S.gguf.part2of2 > OR1-Behemoth.Q5_K_S.gguf
```

**Recommendation**: Test Q4_K_M first to establish baseline performance. If throughput is <5 tokens/second, consider switching to Q5_K_S or Q6_K.

## Troubleshooting

### If llama-server fails to start:
```bash
# Check logs
cat /tmp/server.log

# Verify RPC connectivity
ssh root@10.0.0.31 "ping -c 3 10.100.0.32 && ping -c 3 10.100.0.33"

# Check if RPC servers are listening
ssh root@10.0.0.32 "ss -tlnp | grep 50052"
ssh root@10.0.0.33 "ss -tlnp | grep 50052"
```

### If RPC servers crash:
```bash
# Check logs
ssh root@10.0.0.32 "cat /tmp/rpc.log"
ssh root@10.0.0.33 "cat /tmp/rpc.log"

# Restart
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
nohup /opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc.log 2>&1 &
```

### If download fails:
```bash
# Resume with -c flag
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q4_K_M.gguf" -O OR1-Behemoth.Q4_K_M.gguf
```

## Next Steps

1. ✅ Model download complete
2. Launch llama-server on node1 with RPC backends (using InfiniBand IPs)
3. Test inference via OpenAI-compatible API
4. Verify distributed GPU usage (all 3 V100S should be active)
5. Measure performance and consider switching to Q5_K_S or Q6_K if Q4 is too slow
6. Performance tuning if needed (batch size, context size, etc.)

## Notes

- All 3 nodes have llama.cpp working with CUDA
- RPC servers are ready and listening (0.0.0.0:50052)
- Model download complete: 47.4 GB OR1-Behemoth.Q4_K_M.gguf
- Disk space: 77GB free on node1
- InfiniBand network available for fast inter-node communication
- **Q4 Performance Note**: V100 lacks INT4 tensor cores; expect slower throughput (8-15 tokens/sec)
- **Last Status Check**: 19:04 (January 16, 2026) - download complete

## Quick Reference Paths

```bash
# Node1 (head)
/opt/llama.cpp/build/bin/llama-server      # Main inference server
/opt/llama.cpp/build/bin/llama-cli         # CLI for testing
/opt/models/OR1-Behemoth.Q4_K_M.gguf       # Model file
/tmp/server.log                             # Server logs
/tmp/download.log                          # Download logs

# Node2 (RPC worker)
/opt/llama.cpp/build/bin/rpc-server        # RPC worker
/tmp/rpc.log                                # RPC server logs

# Node3 (RPC worker)
/opt/llama.cpp/build/bin/rpc-server        # RPC worker
/tmp/rpc.log                                # RPC server logs
```

## Environment Variables

```bash
# Required on all nodes before running binaries
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH

# Optional: CUDA path (already configured in PATH)
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
```

## Process Management

### Start RPC Servers (Nodes 2 & 3)
```bash
# Node2
ssh -J root@100.127.208.104 root@10.0.0.32 'bash -s' <<'EOF'
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
nohup /opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc.log 2>&1 &
echo $! > /tmp/rpc.pid
EOF

# Node3
ssh -J root@100.127.30.114 root@10.0.0.33 'bash -s' <<'EOF'
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
nohup /opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc.log 2>&1 &
echo $! > /tmp/rpc.pid
EOF
```

### Stop RPC Servers
```bash
# Node2
ssh -J root@100.127.208.104 root@10.0.0.32 'kill $(cat /tmp/rpc.pid) && rm /tmp/rpc.pid'

# Node3
ssh -J root@100.127.30.114 root@10.0.0.33 'kill $(cat /tmp/rpc.pid) && rm /tmp/rpc.pid'
```

### Start llama-server (Node1)
```bash
ssh -J root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
cd /opt/llama.cpp/build/bin

nohup ./llama-server \
    -m /opt/models/OR1-Behemoth.Q4_K_M.gguf \
    --rpc 10.100.0.32:50052,10.100.0.33:50052 \
    -ngl 999 \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 4096 \
    --n-gpu-layers 999 \
    --parallel 3 \
    --threads 40 \
    > /tmp/server.log 2>&1 &

echo $! > /tmp/server.pid
EOF
```

### Stop llama-server (Node1)
```bash
ssh -J root@100.127.208.104 root@10.0.0.31 'kill $(cat /tmp/server.pid) && rm /tmp/server.pid'
```

## llama.cpp RPC Behavior

### How Distributed Inference Works
- **Layer Splitting**: llama.cpp automatically distributes model layers across available RPC backends
- **Load Balancing**: 72B model Q4 = 48GB → 3 nodes × 16GB each (fits comfortably)
- **Data Flow**:
  1. Node1 receives inference request
  2. Loads model and distributes layers to RPC workers
  3. During forward pass, layers execute on respective nodes
  4. Results aggregated back to node1 for response

### Key Parameters Explained
```bash
--rpc 10.100.0.32:50052,10.100.0.33:50052  # Comma-separated RPC backends
-ngl 999                                      # Offload all layers to GPU (999 = all)
--parallel 3                                  # Parallel processing with 3 GPUs
--ctx-size 4096                               # Context window size
--threads 40                                  # CPU threads per node
--host 0.0.0.0                                # Listen on all interfaces
--port 8000                                   # OpenAI-compatible API port
```

## Verifying Distributed GPU Usage

### Check GPU Utilization During Inference
```bash
# Run inference request in background
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "OR1-Behemoth", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}' > /dev/null 2>&1 &

# Monitor all 3 GPUs simultaneously
watch -n 1 'for node in 31 32 33; do
  echo "=== 10.0.0.$node ==="
  ssh -J root@100.127.208.104 root@10.0.0.$node "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
done'
```

### Expected Behavior
- **Node1**: GPU shows activity (local layers + orchestration)
- **Node2**: GPU shows activity (RPC worker layers)
- **Node3**: GPU shows activity (RPC worker layers)
- All 3 GPUs should have non-zero utilization during inference

## Common Errors & Solutions

### Error: "Connection refused" when starting llama-server
**Cause**: RPC servers not running or not reachable
**Solution**:
```bash
# Check RPC servers
ssh root@10.0.0.32 "ss -tlnp | grep 50052"
ssh root@10.0.0.33 "ss -tlnp | grep 50052"

# Test connectivity from node1
ssh root@10.0.0.31 "ping -c 2 10.100.0.32 && ping -c 2 10.100.0.33"

# Check firewall rules
for node in 31 32 33; do
  ssh -J root@100.127.208.104 root@10.0.0.$node "firewall-cmd --list-all | grep 50052"
done
```

### Error: "CUDA out of memory"
**Cause**: Model too large for available VRAM or context window too big
**Solution**:
```bash
# Check GPU memory
nvidia-smi

# Reduce context size
./llama-server --ctx-size 2048  # instead of 4096

# Or reduce n-gpu-layers
./llama-server --n-gpu-layers 50  # offload fewer layers
```

### Error: "RPC backend disconnected"
**Cause**: Network issue or RPC server crashed
**Solution**:
```bash
# Check RPC server logs
ssh root@10.0.0.32 "tail -50 /tmp/rpc.log"
ssh root@10.0.0.33 "tail -50 /tmp/rpc.log"

# Restart RPC servers (see Process Management section)
```

### Slow Inference Performance
**Cause**: Network bottleneck or suboptimal configuration
**Solution**:
```bash
# Verify InfiniBand is being used (not Ethernet)
ssh root@10.0.0.31 "ip route get 10.100.0.32"  # Should use 10.100.0.x subnet

# Check IB MTU
ssh root@10.0.0.31 "ibstat | grep -A2 State"

# Tune parallelism
./llama-server --parallel 3 --threads 40  # Adjust based on workload
```

## Performance Expectations

### Theoretical Throughput
- **V100S FP16**: ~125 TFLOPS (tensor cores)
- **Q4 De-quantization Penalty**: ~2-3x slower than FP16
- **Inter-node Latency**: ~1-2µs via InfiniBand
- **Expected Speed**: ~8-15 tokens/second for 72B model (varies by prompt complexity)

### Bottlenecks to Monitor
1. **Network IB**: Should sustain >50 Gbps during inference
2. **GPU Memory**: Keep <28GB used (headroom for KV cache)
3. **CPU Threads**: 40 threads/node should be sufficient
4. **Context Window**: Larger context = more VRAM for KV cache

## Alternative Models

If OR1-Behemoth doesn't work well, consider these alternatives:

| Model | Parameters | Quantization | Size | Notes |
|-------|-----------|--------------|------|-------|
| Qwen2.5-72B | 72B | Q4_K_M | 43 GB | Similar architecture, well-tested |
| DeepSeek-V3 | 67B | Q4_K_M | 40 GB | MoE architecture, different RPC behavior |
| Llama-3.1-70B | 70B | Q4_K_M | 42 GB | Llama-native, good llama.cpp support |

**Download alternative model**:
```bash
cd /opt/models
wget "https://huggingface.co/mradermacher/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
```
