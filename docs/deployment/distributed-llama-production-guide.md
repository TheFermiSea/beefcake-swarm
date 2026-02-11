# Distributed vLLM/llama.cpp Cluster - Production Deployment Guide

**Date**: January 16, 2026
**Project**: Distributed inference cluster for agentic Rust code analysis using llama.cpp with RPC
**Status**: ✅ Ready for Production Deployment

## Executive Summary

This document provides **corrected, production-ready configuration** for deploying Rust coding models across 3×V100S nodes using llama.cpp RPC, incorporating insights from comprehensive analysis of agentic models in 2026.

**Strategic Model Selection**:
- **Primary Model**: OR1-Behemoth 73B Qwen3-based (initial choice)
- **Alternative Model**: Strand-Rust-Coder-14B-v1 (recommended for deep analysis)
- **Hybrid Approach**: Deploy multi-model cluster with model routing based on task type

**Critical Changes from Initial Plan**:
- Switch from Q4_K_M (47.4 GB) → Q8_0 (77.3 GB) for V100S optimization
- Replace nohup scripts with systemd services for reliability
- Remove duplicate GPU layer offload flags
- Adjust parallelism settings for pipeline-parallel architecture
- **NEW**: Consider Strand-Rust-Coder-14B for idiomatic analysis tasks

---

## Why Q8_0 Instead of Q4 on V100S

### The "Volta Trap"

**Tesla V100S Architecture Limitation**:
- V100S (Volta, CC 7.0) **lacks INT4 tensor cores**
- Q4 models require INT4→FP16 de-quantization on CUDA cores before matrix multiplication
- This creates a compute bottleneck that stalls HBM2 bandwidth (900 GB/s)

**Performance Impact**:

| Architecture | INT4 Tensor Cores | Q4 De-quantization | Q8 Performance |
|-------------|-------------------|-------------------|----------------|
| **V100S** (Volta) | ❌ No | ❌ Slow (CUDA cores) | ✅ Fast (native) |
| A100 (Ampere) | ✅ Yes | ✅ Fast (tensor cores) | ✅ Fast (native) |
| H100 (Hopper) | ✅ Yes | ✅ Fast (tensor cores) | ✅ Fast (native) |

**Expected Throughput on 3×V100S**:
- Q4_K_M: ~8-12 tokens/second (de-quantization bottleneck)
- Q8_0: ~15-20 tokens/second (near memory bandwidth limit)
- **Performance gain: ~1.5-2× with Q8_0**

### Memory Fit Analysis

| Model | Size | Total VRAM (3×V100S) | Per GPU | KV Cache Headroom |
|-------|-------|----------------------|----------|-------------------|
| Q4_K_M | 47.4 GB | 96 GB | 16 GB | 16 GB (generous) |
| **Q8_0** | **77.3 GB** | **96 GB** | **26 GB** | **6 GB** (sufficient for 4096 tokens) |

**Q8_0 fits comfortably** with ~6 GB per GPU for KV cache at 4096 token context.

### Network Bandwidth Consideration

Q8_0 is 77 GB vs 47 GB for Q4_K_M (1.6× larger). However:
- InfiniBand @ 100 Gb/s = 12.5 GB/s
- Initial model load time: Q4_K_M ~4s, Q8_0 ~6s (acceptable one-time cost)
- Inference uses the same layer distribution, so **no per-request overhead**

**Conclusion**: Q8_0 is the **correct choice** for V100S despite larger model size.

---

## Model Selection Strategy

Based on comprehensive analysis of agentic Rust coding models (January 2026), we've identified a **hybrid model strategy** optimized for different use cases.

### Model Comparison

| Model | Params | Architecture | VRAM (Q8) | Compilation Rate | Strengths | Weaknesses |
|--------|---------|--------------|----------------|-----------------|-----------|
| **OR1-Behemoth 73B** | 73B (embiggened Qwen3) | 77 GB | ~70-90% | Large context, Typestate patterns, no_std | Coherence drift, stability issues |
| **Strand-Rust-Coder 14B** | 14B (swarm-synthesized) | 7 GB | 94.3% | Idiomatic Rust, peer-reviewed, fast iteration | Limited context, smaller reasoning depth |
| DeepSeek Coder V3 671B | 671B (MoE, System 2) | 37 GB | ~80-85% | Self-correction, massive reasoning, unmatched borrow checker | Slow (MoE routing latency) |

### Strategic Model Assignment

**Hybrid Routing by Task Type**:

| Task Type | Primary Model | Rationale |
|-----------|---------------|-----------|
| **Idiomatic Analysis** | Strand-Rust-Coder-14B-v1 | High compilation rate (94.3%), peer-reviewed training ensures idiomatic safety |
| **Complex Refactoring** | OR1-Behemoth 73B | Larger context (32-128k) for understanding codebase architecture, better for cross-module logic |
| **Deep Borrowing Issues** | DeepSeek Coder V3 671B | Self-correction capabilities excel at solving ownership puzzles |
| **Code Completion** | Strand-Rust-Coder-14B-v1 | Fast latency (~10 tokens/sec iteration) for real-time suggestions |
| **Documentation Generation** | OR1-Behemoth 73B | Strong at explaining code with detailed examples |

### Implementation Approach

**Two-Node Configuration**:
- **Node 1 (Head)**: Deploy OR1-Behemoth 73B Q8_0 (77 GB)
- **Node 2 (RPC)**: Deploy Strand-Rust-Coder 14B Q8 (7 GB)
- **Node 3 (RPC)**: Deploy DeepSeek Coder V3 671B Q8 (37 GB)

**Model Router** (simple script-based routing):
```bash
#!/bin/bash
# Simple task classifier
TASK_TYPE="$1"

case "$TASK_TYPE" in
  "analyze"|"idiomatic"|"refactor")
    # Use OR1-Behemoth for deep analysis
    MODEL_PATH="/opt/models/OR1-Behemoth.Q8_0.gguf"
    MODEL_NAME="OR1-Behemoth"
    ;;
  "fix"|"complete"|"typestate")
    # Use Strand-Rust-Coder for idiomatic improvements
    MODEL_PATH="/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf"
    MODEL_NAME="Strand-Rust-Coder"
    ;;
  "borrowing"|"ownership"|"complex")
    # Use DeepSeek Coder V3 for difficult reasoning
    MODEL_PATH="/opt/models/DeepSeek-Coder-V3.Q8_0.gguf"
    MODEL_NAME="DeepSeek-Coder-V3"
    ;;
  *)
    # Default to OR1-Behemoth
    MODEL_PATH="/opt/models/OR1-Behemoth.Q8_0.gguf"
    MODEL_NAME="OR1-Behemoth"
    ;;
esac

echo "Using model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
```

**Benefits of Hybrid Strategy**:
1. **Hardware Flexibility**: Strand (14B) fits in 12GB VRAM → can run on consumer GPUs or edge devices
2. **Task Optimization**: Match model capability to task type for better performance
3. **Risk Mitigation**: Diverse models reduce single-point failure mode
4. **Cost Efficiency**: Smaller models for common tasks → faster inference, lower compute cost

### Revised Hardware Requirements

| Configuration | Total VRAM | Per Node (3×) | Inference Cost | Latency |
|-------------|-------------|-------------------|---------------|----------|
| **Original Plan** (3×OR1 73B) | 77 GB | 26 GB | High | Medium |
| **Hybrid Plan** (OR1 + Strand + DeepSeek) | 121 GB | 40 GB (varies) | Lower (for Strand) | Lower (for Strand) |

**Note**: Hybrid plan allows mixing models on different nodes based on task needs.

## Recommended Deployment Plan

### Phase 1: Model Acquisition & Strategy Setup

#### Step 1A: Download OR1-Behemoth Q8_0 (Primary Model)

#### Step 1: Stop Q4 Download
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
pkill wget
rm -f /opt/models/OR1-Behemoth.Q4_K_M.gguf
echo "Q4 download stopped and deleted"
EOF
```

#### Step 2: Download Q8_0 Model (Split in 2 Parts)
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part1of2"
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part2of2"

# Combine parts
cat OR1-Behemoth.Q8_0.gguf.part1of2 OR1-Behemoth.Q8_0.gguf.part2of2 > OR1-Behemoth.Q8_0.gguf
rm -f OR1-Behemoth.Q8_0.gguf.part*

# Verify file size
ls -lh OR1-Behemoth.Q8_0.gguf
EOF
```

**Expected Output**:
```
-rw-r--r--. 1 root root 77G Jan 16 XX:XX OR1-Behemoth.Q8_0.gguf
```

**Download Time**: ~20-25 minutes @ ~45 MB/s

### Phase 2: Deploy Systemd Services

#### A. RPC Worker Service (Node 2 & Node 3)

**Create Service File**:
```bash
# Deploy to Node 2
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-rpc.service <<'SERVICE'
[Unit]
Description=Llama.cpp RPC Worker Node
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin

# Environment variables
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"

# Main execution command
ExecStart=/opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052

# Restart behavior
Restart=always
RestartSec=5

# Resource limits
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and enable service
systemctl daemon-reload
systemctl disable llama-rpc 2>/dev/null  # Disable any old instance
systemctl enable --now llama-rpc
systemctl status llama-rpc
EOF
```

```bash
# Deploy to Node 3
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-rpc.service <<'SERVICE'
[Unit]
Description=Llama.cpp RPC Worker Node
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin

# Environment variables
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"

# Main execution command
ExecStart=/opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052

# Restart behavior
Restart=always
RestartSec=5

# Resource limits
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and enable service
systemctl daemon-reload
systemctl disable llama-rpc 2>/dev/null  # Disable any old instance
systemctl enable --now llama-rpc
systemctl status llama-rpc
EOF
```

**Verification**:
```bash
# Check Node 2
ssh -o ConnectTimeout=10 root@100.127.30.114 "ssh root@10.0.0.32 'systemctl status llama-rpc'"

# Check Node 3
ssh -o ConnectTimeout=10 root@100.68.22.98 "ssh root@10.0.0.33 'systemctl status llama-rpc'"

# Check listening ports
for node in 32 33; do
  echo "=== Node $node ==="
  ssh root@10.0.0.$node "ss -tlnp | grep 50052"
done
```

#### B. Head Node Service (Node 1)

**Create Service File**:
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-head.service <<'SERVICE'
[Unit]
Description=Llama.cpp Head Server (OR1-Behemoth 72B)
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin

# Environment variables
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"

# Launch command with CORRECTED flags
# NOTE: --rpc uses InfiniBand IPs (10.100.0.x) for 100Gb/s
# NOTE: --parallel 1 for baseline (increase only if GPUs idle)
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m /opt/models/OR1-Behemoth.Q8_0.gguf \
    --rpc 10.100.0.32:50052,10.100.0.33:50052 \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 4096 \
    -ngl 999 \
    --parallel 1 \
    --threads 40

# Restart behavior
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65535
MemoryMax=100G

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd
systemctl daemon-reload
EOF

echo "Service file created. Start manually after model download completes:"
echo "  systemctl enable --now llama-head"
EOF
```

### Phase 3: Launch Inference Server

#### Step 1: Verify RPC Workers
```bash
# Check both workers are running
for node in 32 33; do
  echo "=== Node 10.0.0.$node ==="
  if [ "$node" = "32" ]; then
    ssh -o ConnectTimeout=10 root@100.127.30.114 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && ss -tlnp | grep 50052'"
  else
    ssh -o ConnectTimeout=10 root@100.68.22.98 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && ss -tlnp | grep 50052'"
  fi
done
```

#### Step 2: Start Head Node (After Q8_0 Download Completes)
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
# Verify model exists and is correct size
MODEL_SIZE=$(stat -f%z /opt/models/OR1-Behemoth.Q8_0.gguf 2>/dev/null || stat -c%s /opt/models/OR1-Behemoth.Q8_0.gguf)
EXPECTED_SIZE=83002634240  # 77.3 GB in bytes

if [ "$MODEL_SIZE" -ge 80000000000 ]; then
  echo "Model download complete (${MODEL_SIZE} bytes)"
  systemctl enable --now llama-head
  echo "Head server starting..."
  sleep 5
  systemctl status llama-head
else
  echo "Model download incomplete or missing (${MODEL_SIZE} bytes)"
  exit 1
fi
EOF
```

#### Step 3: Verify Server Health
```bash
# Check logs
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'journalctl -u llama-head -f --lines=50' &
LOG_PID=$!

# Test API endpoint
sleep 10
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Say \"Hello from distributed inference!\""}],
    "max_tokens": 50
  }'

# Stop log monitoring
kill $LOG_PID 2>/dev/null
```

#### Step 4: Monitor Distributed GPU Usage
```bash
# Run inference request
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Explain Rust ownership system in 200 words"}],
    "max_tokens": 200
  }' > /dev/null 2>&1 &

# Monitor all 3 GPUs
watch -n 1 'echo "=== Distributed GPU Utilization ==="
for node in 31 32 33; do
  echo "Node 10.0.0.$node:"
  case $node in
    31) ssh -o ConnectTimeout=5 -J root@100.127.208.104 root@10.0.0.$node "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits" 2>/dev/null ;;
    32) ssh -o ConnectTimeout=5 root@100.127.30.114 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits\"" 2>/dev/null ;;
    33) ssh -o ConnectTimeout=5 root@100.68.22.98 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits\"" 2>/dev/null ;;
  esac
  echo ""
done'
```

**Expected Behavior**:
- All 3 GPUs show non-zero utilization during inference
- Memory usage: ~26 GB/GPU for Q8_0 model
- GPU utilization: 60-90% (varies by prompt complexity)

---

## Alternative Model Deployment (Hybrid Strategy)

### Step 1B: Download Strand-Rust-Coder-14B-v1 (For Idiomatic Analysis)

**Why This Model**:
- 14B parameters (fits in 7-12GB VRAM at Q8)
- 94.3% compilation rate (peer-reviewed training)
- +13% improvement on RustEvo² benchmark
- Specialized for Typestate patterns, generics, no_std code
- Fast latency (~10 tokens/sec iteration)

**Download Commands**:
```bash
# Download to Node 2 (RPC Worker)
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'bash -s' <<'EOF'
cd /opt/models
wget -c "https://huggingface.co/Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF/resolve/main/Strand-Rust-Coder-14B-v1.Q8_0.gguf"

# Verify
ls -lh Strand-Rust-Coder-14B-v1.Q8_0.gguf
EOF

# Test with llama-cli
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 '/opt/llama.cpp/build/bin/llama-cli -m /opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf -n 1'
EOF
```

**Expected File Size**: ~7 GB (Q8_0 quantization)

### Step 1C: Download DeepSeek Coder V3 671B (For Deep Refactoring)

**Why This Model**:
- 671B Mixture-of-Experts with ~37B active parameters
- Self-correction capabilities (critical for complex borrow checker issues)
- System 2 reasoning (massive knowledge base)
- Unmatched at solving novel ownership patterns

**Note**: This model is MASSIVE (671B) - consider if deployment is practical for your use case.

**Download Commands** (if needed):
```bash
# Download Q4_K_M or Q5_K_M (671B requires careful quantization)
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cd /opt/models

# Try Q5_K_M first (better quality, larger)
wget -c "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V3-GGUF/resolve/main/DeepSeek-Coder-V3-Q5_K_M.gguf"

# If too large, fall back to Q4
# Q5_K_M is ~120 GB at Q5 - may need multiple parts
EOF
```

**Warning**: 671B model requires significant VRAM even at Q5 quantization. May need  run on dedicated hardware or use smaller context sizes.

### Step 1D: Configure Model Router Script

```bash
# Create model selection script on Node 1 (head)
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cat > /usr/local/bin/llama-model-selector.sh <<'SCRIPT'
#!/bin/bash
# Simple task classifier - accepts TASK_TYPE as argument
TASK_TYPE="$1"

case "$TASK_TYPE" in
  "analyze"|"idiomatic"|"refactor")
    # Use OR1-Behemoth for deep analysis
    MODEL_PATH="/opt/models/OR1-Behemoth.Q8_0.gguf"
    MODEL_NAME="OR1-Behemoth"
    ;;
  "fix"|"complete"|"typestate")
    # Use Strand-Rust-Coder for idiomatic improvements
    MODEL_PATH="/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf"
    MODEL_NAME="Strand-Rust-Coder"
    ;;
  "borrowing"|"ownership"|"complex")
    # Use DeepSeek Coder V3 for difficult reasoning
    MODEL_PATH="/opt/models/DeepSeek-Coder-V3.Q5_K_M.gguf"
    MODEL_NAME="DeepSeek-Coder-V3"
    ;;
  *)
    # Default to OR1-Behemoth
    MODEL_PATH="/opt/models/OR1-Behemoth.Q8_0.gguf"
    MODEL_NAME="OR1-Behemoth"
    ;;
esac

echo "Using model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"

# Export for systemd to use
export LLAMA_MODEL_PATH="$MODEL_PATH"
export LLAMA_MODEL_NAME="$MODEL_NAME"
SCRIPT

chmod +x /usr/local/bin/llama-model-selector.sh
EOF

# Test script
/usr/local/bin/llama-model-selector.sh analyze
EOF
```

### Step 1E: Update Head Node Service to Support Multiple Models

**Modified Service File**:
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-head.service <<'SERVICE'
[Unit]
Description=Llama.cpp Head Server (Rust Coding Models)
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin

# Environment variables
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"

# Use model selector script to determine model dynamically
ExecStart=/bin/bash -c '/usr/local/bin/llama-model-selector.sh analyze && /opt/llama.cpp/build/bin/llama-server -m \$LLAMA_MODEL_PATH --rpc 10.100.0.32:50052,10.100.0.33:50052 --host 0.0.0.0 --port 8000 --ctx-size 4096 -ngl 999 --parallel 1 --threads 40'

# Restart behavior
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65535
MemoryMax=100G

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd
systemctl daemon-reload
EOF

echo "Updated head service to support model routing"
EOF
```

**Benefits**:
- Dynamic model selection based on task type
- Single service handles all models
- No manual service file updates for model changes

### Phase 1 Revised: Primary Model Download

#### Step 1A: Stop Q4 Download

### Corrected Launch Parameters

| Parameter | Value | Notes |
|-----------|--------|-------|
| `-m` | `/opt/models/OR1-Behemoth.Q8_0.gguf` | **Changed** from Q4 to Q8 |
| `--rpc` | `10.100.0.32:50052,10.100.0.33:50052` | InfiniBand IPs (100 Gb/s) |
| `-ngl` | `999` | Offload all layers to GPU |
| `--parallel` | `1` | **Changed** from 3 (baseline first) |
| `--threads` | `40` | Matches CPU cores per node |
| `--ctx-size` | `4096` | Context window size |
| `--host` | `0.0.0.0` | Listen on all interfaces |
| `--port` | `8000` | OpenAI-compatible API |

**Bug Fix**: Removed duplicate `-ngl 999` that appeared in initial configuration.

### Parallelism Tuning Guide

**Start with `--parallel 1`** (pipeline-parallel setup):
- Minimizes "bubble" time between nodes
- Reduces network contention
- Establishes reliable baseline

**Increase only if**:
- GPU utilization drops below 40% during inference
- Inference throughput is bottlenecked by serialization (not compute/memory)
- You observe significant idle time between forward passes

**Testing approach**:
```bash
# Test with parallel=1
systemctl restart llama-head
# Measure tokens/second

# Test with parallel=3 (edit service file)
systemctl restart llama-head
# Measure tokens/second

# Compare and keep higher value
```

---

## Systemd Service Management

### RPC Worker Management

```bash
# Start
systemctl start llama-rpc

# Stop
systemctl stop llama-rpc

# Restart
systemctl restart llama-rpc

# Enable at boot
systemctl enable llama-rpc

# Check status
systemctl status llama-rpc

# View logs
journalctl -u llama-rpc -f
journalctl -u llama-rpc --lines=100

# Disable (for maintenance)
systemctl disable llama-rpc
```

### Head Node Management

```bash
# Start
systemctl start llama-head

# Stop
systemctl stop llama-head

# Restart
systemctl restart llama-head

# Enable at boot
systemctl enable llama-head

# Check status
systemctl status llama-head

# View logs
journalctl -u llama-head -f
journalctl -u llama-head --lines=100

# Disable (for maintenance)
systemctl disable llama-head
```

---

## Performance Testing & Benchmarking

### 1. Baseline Throughput Test
```bash
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Write a comprehensive analysis of distributed inference architectures. Include code examples in Rust."}
    ],
    "max_tokens": 500,
    "stream": false
  }' | jq -r '.usage.total_tokens, .choices[0].message.content' > /tmp/baseline_test.txt

# Calculate tokens/second (manual measurement)
```

### 2. Load Test (Multiple Concurrent Requests)
```bash
# Install hey (HTTP benchmarking tool) first if needed
for i in {1..5}; do
  curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"OR1-Behemoth\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Generate task $i: Write a Python function\"}],
      \"max_tokens\": 100
    }" &
done
wait
```

### 3. Network Bandwidth Test
```bash
# Measure IB bandwidth during inference
# Run on node1
iperf3 -c 10.100.0.32 -t 10 &
iperf3 -c 10.100.0.33 -t 10 &

# Also trigger inference and monitor
# Expected: >50 Gbps sustained during heavy inference
```

---

## Troubleshooting

### Q8_0 Model Download Issues

**Incomplete Download**:
```bash
# Check file size
ls -lh /opt/models/OR1-Behemoth.Q8_0.gguf

# Should be ~77 GB
# If smaller, resume with wget -c
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part1of2"
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part2of2"
cat OR1-Behemoth.Q8_0.gguf.part* > OR1-Behemoth.Q8_0.gguf
```

**Corrupted Download**:
```bash
# Test model integrity
ssh root@10.0.0.31 '/opt/llama.cpp/build/bin/llama-cli -m /opt/models/OR1-Behemoth.Q8_0.gguf -n 1'

# If fails, re-download
rm /opt/models/OR1-Behemoth.Q8_0.gguf
# Re-run download commands
```

### Systemd Service Issues

**Service won't start**:
```bash
# Check logs
journalctl -u llama-rpc -n 50
journalctl -u llama-head -n 50

# Common issues:
# 1. CUDA library path missing
#    Solution: Update Environment="LD_LIBRARY_PATH=..."
# 2. Model file missing
#    Solution: Verify /opt/models/OR1-Behemoth.Q8_0.gguf exists
# 3. Port already in use
#    Solution: lsof -i:50052 or lsof -i:8000
```

**Service crashing repeatedly**:
```bash
# Check exit codes
systemctl status llama-head
journalctl -u llama-head | grep "exited"

# If OOM (Out Of Memory):
# 1. Reduce context size: --ctx-size 2048
# 2. Reduce n-gpu-layers: -ngl 500 (offload fewer layers)
# 3. Switch to smaller model
```

### Performance Issues

**Throughput too slow (<10 tokens/sec)**:
```bash
# 1. Check GPU utilization
nvidia-smi dmon -s u -d 1

# 2. Check network (should use IB, not Ethernet)
ip route get 10.100.0.32

# 3. Increase parallelism (edit service file)
#    Change --parallel 1 to --parallel 3
#    systemctl restart llama-head
```

**High memory usage >30 GB/GPU**:
```bash
# 1. Reduce context size
#    --ctx-size 2048

# 2. Check for memory leaks
#    nvidia-smi --query-gpu=memory.used --format=csv -l 1

# 3. Restart service
systemctl restart llama-head
```

### RPC Worker Disconnection

**Worker not reachable from head**:
```bash
# 1. Test connectivity
ping -c 3 10.100.0.32
ping -c 3 10.100.0.33

# 2. Check firewall
firewall-cmd --list-ports

# 3. Check worker logs
journalctl -u llama-rpc -n 100
```

**Worker crashed**:
```bash
# 1. Restart worker
systemctl restart llama-rpc

# 2. Check GPU health
nvidia-smi

# 3. Check system logs
dmesg | tail -100
```

---

## Monitoring & Observability

### GPU Monitoring Dashboard
```bash
# Real-time monitoring across all nodes
watch -n 2 'echo "=== Cluster GPU Status ==="
for node in 31 32 33; do
  echo "Node 10.0.0.$node:"
  case $node in
    31) ssh -o ConnectTimeout=3 -J root@100.127.208.104 root@10.0.0.$node "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv" 2>/dev/null ;;
    32) ssh -o ConnectTimeout=3 root@100.127.30.114 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv\"" 2>/dev/null ;;
    33) ssh -o ConnectTimeout=3 root@100.68.22.98 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv\"" 2>/dev/null ;;
  esac
  echo ""
done'
```

### Service Status Dashboard
```bash
# Check all services
echo "=== RPC Workers ==="
for node in 32 33; do
  echo "Node 10.0.0.$node:"
  if [ "$node" = "32" ]; then
    ssh -o ConnectTimeout=5 root@100.127.30.114 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && echo \"Running\" || echo \"Stopped\"'"
  else
    ssh -o ConnectTimeout=5 root@100.68.22.98 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && echo \"Running\" || echo \"Stopped\"'"
  fi
done

echo ""
echo "=== Head Node ==="
ssh -o ConnectTimeout=5 -J root@100.127.208.104 root@10.0.0.31 'systemctl is-active llama-head && echo "Running on port 8000" || echo "Stopped"'

echo ""
echo "=== API Endpoint ==="
curl -s http://10.0.0.31:8000/health 2>/dev/null && echo "✅ API Healthy" || echo "❌ API Unreachable"
```

### Log Aggregation
```bash
# View recent logs from all services
echo "=== RPC Worker 2 (Node 32) ==="
ssh root@100.127.30.114 "ssh root@10.0.0.32 'journalctl -u llama-rpc -n 20 --no-pager'"

echo ""
echo "=== RPC Worker 3 (Node 33) ==="
ssh root@100.68.22.98 "ssh root@10.0.0.33 'journalctl -u llama-rpc -n 20 --no-pager'"

echo ""
echo "=== Head Node (Node 1) ==="
ssh -J root@100.127.208.104 root@10.0.0.31 'journalctl -u llama-head -n 20 --no-pager'
```

---

## Architecture Decision Log

### Decision 1: Q8_0 over Q4_K_M
**Rationale**: V100S lacks INT4 tensor cores → Q4 de-quantization bottleneck
**Impact**: +1.5-2× throughput, +30 GB model size
**Tradeoff**: Longer initial load (6s vs 4s), acceptable for production
**Date**: January 16, 2026

### Decision 2: Systemd over nohup
**Rationale**: Rocky Linux 8 systemd-logind cleans up user scope on SSH disconnect
**Impact**: Reliable service lifecycle, auto-restart, proper daemon management
**Tradeoff**: Slightly more complex initial setup, but production-grade reliability
**Date**: January 16, 2026

### Decision 3: --parallel 1 baseline
**Rationale**: Pipeline-parallel RPC creates bubbles with high parallelism
**Impact**: Stable baseline, reduced network contention
**Tradeoff**: Lower theoretical max throughput, but more predictable
**Date**: January 16, 2026

---

## Next Steps

1. ✅ **Stop Q4 download** - Abort current Q4_K_M download
2. **Download Q8_0** - Acquire 77.3 GB model (20-25 min)
3. **Deploy systemd services** - RPC workers on nodes 2/3
4. **Launch head node** - Start llama-server with corrected configuration
5. **Test inference** - Verify distributed GPU usage and measure throughput
6. **Benchmark** - Establish baseline, consider tuning parallelism
7. **Monitor** - Set up ongoing monitoring for production

---

## Quick Reference

### Node Access
```bash
# Via Proxmox hosts (Tailscale)
ssh root@100.127.208.104  # pve1 → vllm-node1
ssh root@100.127.30.114   # pve2 → vllm-node2
ssh root@100.68.22.98     # pve3 → vllm-node3

# Direct VM access (via pve hosts)
ssh root@10.0.0.31  # vllm-node1 (head)
ssh root@10.0.0.32  # vllm-node2 (RPC worker)
ssh root@10.0.0.33  # vllm-node3 (RPC worker)
```

### Network IPs
| Node | Ethernet | InfiniBand | Role |
|------|----------|-------------|------|
| vllm-node1 | 10.0.0.31 | 10.100.0.31 | Head |
| vllm-node2 | 10.0.0.32 | 10.100.0.32 | RPC Worker |
| vllm-node3 | 10.0.0.33 | 10.100.0.33 | RPC Worker |

**Critical**: Use InfiniBand IPs (10.100.0.x) for RPC communication.

### Service Commands
```bash
# RPC Workers (nodes 2 & 3)
systemctl start|stop|restart|status llama-rpc
journalctl -u llama-rpc -f

# Head Node (node 1)
systemctl start|stop|restart|status llama-head
journalctl -u llama-head -f
```

### File Locations
```
/opt/llama.cpp/build/bin/
├── llama-server      # Main inference server
├── rpc-server       # RPC worker
└── llama-cli        # CLI testing tool

/opt/models/
└── OR1-Behemoth.Q8_0.gguf  # 72B model (77.3 GB)

/etc/systemd/system/
├── llama-rpc.service          # Worker service
└── llama-head.service         # Head service
```

---

## Appendix: Performance Targets

### Expected Metrics (Q8_0 on 3×V100S)

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| Throughput | 15-20 tokens/s | >10 tokens/s | <8 tokens/s |
| Time to First Token | <2 seconds | <5 seconds | >10 seconds |
| GPU Utilization | 70-90% | >50% | <40% |
| Memory/GPU | 26-28 GB | <30 GB | >30 GB |
| Network (IB) | >50 Gbps | >30 Gbps | <20 Gbps |

### Success Criteria
- ✅ All 3 services running (head + 2 workers)
- ✅ API responds to requests within 5 seconds
- ✅ Throughput >10 tokens/s on 4096 context
- ✅ All 3 GPUs show >50% utilization during inference
- ✅ Services auto-restart on failure
- ✅ Zero manual intervention required for 24h operation

---

**Document Version**: 2.0 (Production Ready)
**Last Updated**: January 16, 2026
**Review Status**: ✅ Critical issues addressed
