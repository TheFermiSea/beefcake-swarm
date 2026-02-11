# Next Agent Prompt - Agentic Rust Cluster Deployment

**Project**: Distributed inference cluster for agentic Rust coding using llama.cpp with RPC backend
**Date**: January 16, 2026
**Status**: 📋 Ready for Deployment - Phase 1: Multi-Model Acquisition

---

## Project Overview

You are deploying a production-grade distributed inference cluster for agentic Rust coding tasks. The cluster uses llama.cpp with RPC (Remote Procedure Call) backend to distribute a 72B model across 3×Tesla V100S GPUs connected via InfiniBand @ 100 Gb/s.

### Strategic Decision: Hybrid Multi-Model Approach

Based on comprehensive analysis of 2026 agentic Rust models, we've adopted a **task-optimized multi-model strategy** instead of using a single model (OR1-Behemoth 73B) for all tasks.

**Three Models Deployed**:
1. **OR1-Behemoth 73B Q8_0** (77 GB) - Node 1 (Head)
   - Use for: Deep analysis, architectural refactoring, complex logic
   - Context: 32-128k tokens, large context window
   - Specialization: Typestate patterns, generics, no_std code

2. **Strand-Rust-Coder 14B Q8_0** (7 GB) - Node 2 (RPC Worker)
   - Use for: Idiomatic analysis, real-time suggestions, Typestate refactoring
   - Context: 32k tokens, fast iteration
   - Specialization: 94.3% compilation rate (peer-reviewed swarm training)

3. **DeepSeek Coder V3 671B Q5_K_M** (~120 GB) - Node 3 (RPC Worker)
   - Use for: Complex ownership issues, borrow checker puzzles, novel patterns
   - Context: 32-128k tokens
   - Specialization: Self-correction, massive knowledge base, unmatched reasoning

**Model Router**: Script on head node dynamically selects model based on task type:
- `analyze` → OR1-Behemoth (deep analysis)
- `idiomatic` → Strand-Rust-Coder (idiomatic improvements)
- `borrowing` → DeepSeek Coder (complex ownership)
- `refactor` → Strand-Rust-Coder (idiomatic improvements)
- Other types → Strand-Rust-Coder (idiomatic fixes)

### Infrastructure

| Node | Role | GPU | VRAM | IP (Eth) | IP (IB) |
|-------|-------|-----|-----------|-----------|---------|
| **vllm-node1** | Head | V100S 32GB | 77GB (model) | 10.0.0.31 | 10.100.0.31 |
| **vllm-node2** | RPC Worker | V100S 32GB | 7GB (model) | 10.0.0.32 | 10.100.0.32 |
| **vllm-node3** | RPC Worker | V100S 32GB | ~120GB (model) | 10.0.0.33 | 10.100.0.33 |

**Network**:
- InfiniBand ConnectX-6 HDR100 @ 100 Gb/s (10.100.0.x subnet)
- Must use IB IPs for RPC communication (not Ethernet)

**Software**:
- llama.cpp v7760 built with GCC 12.2.1, CUDA 12.2
- Rocky Linux 8.10
- Systemd services (auto-restart, survive SSH disconnects)

---

## Current State

### ✅ Completed Work

1. **llama.cpp Build** - All 3 nodes have working binaries:
   - `/opt/llama.cpp/build/bin/llama-server` (7 MB)
   - `/opt/llama.cpp/build/bin/rpc-server` (184 KB)
   - `/opt/llama.cpp/build/bin/llama-cli` (5.3 MB)
   - Built with CUDA 12.2, GGML_CUDA=ON, GGML_RPC=ON

2. **Documentation Consolidated** - All docs organized in `docs/agentic-rust-cluster/`:
   - `README.md` - Comprehensive overview (START HERE)
   - `distributed-llama-production-guide.md` - Complete deployment guide
   - `deployment-strategy-update.md` - Strategy comparison
   - `hybrid-model-strategy.md` - Model analysis
   - `beads-epic-summary.md` - Complete Beads epic (17 tasks)
   - `CONSOLIDATION-SUMMARY.md` - Consolidation summary

3. **Beads Epic Created** - `bd show beefcake2-lhr0`:
   - Title: "Distributed OR1-Behemoth 72B Inference Cluster - Production Deployment (Updated with Hybrid Strategy)"
   - Priority: P0 (Highest)
   - Status: All 17 tasks marked as READY (not started)
   - Dependencies configured across 6 phases

4. **Hybrid Strategy Designed** - Based on 2026 agentic Rust model analysis:
   - OR1-Behemoth 73B: Better for deep analysis, but coherence drift issues
   - Strand-Rust-Coder 14B: 94.3% compile rate, idiomatic specialist
   - DeepSeek Coder V3 671B: Best for complex ownership, but massive model

### ⏳ Ready to Start

**Phase 1: Multi-Model Acquisition**
- Status: All 3 download commands ready to execute
- Estimated time: 65-80 minutes (all models)
- Model files to download:
  1. OR1-Behemoth.Q8_0.gguf (77 GB, 2 parts)
  2. Strand-Rust-Coder-14B-v1.Q8_0.gguf (7 GB, single file)
  3. DeepSeek-Coder-V3.Q5_K_M.gguf (~120 GB, 2 parts)

**All subsequent phases** (2-6) are blocked until Phase 1 completes.

---

## Your Mission

### Phase 1: Multi-Model Acquisition (65-80 minutes estimated)

**Objective**: Download all 3 models to their respective nodes with correct quantization.

#### Step 1A: Stop Q4 Download & Download OR1-Behemoth Q8_0

**What to Do**:
1. Stop the running Q4_K_M download (47.4 GB partial file from previous session)
2. Delete the incomplete Q4_K_M file
3. Download OR1-Behemoth Q8_0 model in 2 parts:
   - Part 1: 38.7 GB
   - Part 2: 38.6 GB
4. Combine parts into single GGUF file (77.3 GB total)
5. Verify file size matches expected (~83,026,342,40 bytes)
6. Verify file integrity with `llama-cli`

**Commands** (all on Node 1 - 10.0.0.31):
```bash
# Stop Q4 download
pkill wget
rm -f /opt/models/OR1-Behemoth.Q4_K_M.gguf

# Download Q8_0 parts
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part1of2"
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part2of2"

# Combine parts
cat OR1-Behemoth.Q8_0.gguf.part1of2 OR1-Behemoth.Q8_0.gguf.part2of2 > OR1-Behemoth.Q8_0.gguf
rm -f OR1-Behemoth.Q8_0.gguf.part*

# Verify file size
ls -lh OR1-Behemoth.Q8_0.gguf

# Verify with llama-cli
/opt/llama.cpp/build/bin/llama-cli -m /opt/models/OR1-Behemoth.Q8_0.gguf -n 1
```

**Expected Time**: 20-25 minutes @ ~45 MB/s
**Expected Disk Usage**: ~77 GB download + 38 GB partial = 115 GB temporary (have 77 GB free after cleanup)

#### Step 1B: Download Strand-Rust-Coder 14B Q8_0

**What to Do**:
1. Download Strand-Rust-Coder-14B-v1.Q8_0.gguf (7 GB, single file) to Node 2
2. Verify file size (~7.4 GB)
3. Test with llama-cli

**Commands** (on Node 2 - 10.0.0.32 via pve2):
```bash
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'bash -s' <<'EOF'
cd /opt/models
wget -c "https://huggingface.co/Fortytwo-Network/Strand-Rust-Coder-14B-v1-GGUF/resolve/main/Strand-Rust-Coder-14B-v1.Q8_0.gguf"

# Verify
ls -lh Strand-Rust-Coder-14B-v1.Q8_0.gguf

# Test
/opt/llama.cpp/build/bin/llama-cli -m /opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf -n 1
EOF
```

**Expected Time**: 5 minutes @ ~45 MB/s

#### Step 1C: Download DeepSeek Coder V3 671B Q5_K_M

**What to Do**:
1. Download DeepSeek-Coder-V3.Q5_K_M.gguf in 2 parts:
   - Part 1: ~60 GB
   - Part 2: ~60 GB
2. Combine parts into single GGUF file (~120 GB total)
3. Verify file size
4. Verify with llama-cli

**Commands** (on Node 3 - 10.0.0.33 via pve3):
```bash
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'bash -s' <<'EOF'
cd /opt/models
wget -c "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V3-GGUF/resolve/main/DeepSeek-Coder-V3-Q5_K_M.gguf.part1of2"
wget -c "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V3-GGUF/resolve/main/DeepSeek-Coder-V3-Q5_K_M.gguf.part2of2"

# Combine parts
cat DeepSeek-Coder-V3-Q5_K_M.gguf.part1of2 DeepSeek-Coder-V3-Q5_K_M.gguf.part2of2 > DeepSeek-Coder-V3.Q5_K_M.gguf
rm -f DeepSeek-Coder-V3-Q5_K_M.gguf.part*

# Verify
ls -lh DeepSeek-Coder-V3.Q5_K_M.gguf

# Test with llama-cli (optional, may be too large)
/opt/llama.cpp/build/bin/llama-cli -m /opt/models/DeepSeek-Coder-V3.Q5_K_M.gguf -n 1
EOF
```

**Expected Time**: 40-50 minutes @ ~45 MB/s
**Expected Disk Usage**: ~120 GB (Node 3 has 77 GB free - this will be tight!)

**⚠️ Warning**: DeepSeek V3 is MASSIVE - monitor disk space closely during download. May need to stop Q4 download on Node 3 first or use external storage if insufficient.

#### Step 1D: Verify All Models Downloaded

**What to Do**:
1. Verify all 3 model files exist:
   - Node 1: `/opt/models/OR1-Behemoth.Q8_0.gguf` (~77 GB)
   - Node 2: `/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf` (~7 GB)
   - Node 3: `/opt/models/DeepSeek-Coder-V3.Q5_K_M.gguf` (~120 GB)
2. Verify file sizes match expected values
3. Run quick integrity check with `llama-cli` on smaller models

**Verification Commands**:
```bash
# Check Node 1
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'ls -lh /opt/models/*.gguf && /opt/llama.cpp/build/bin/llama-cli -m /opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf -n 1'

# Check Node 2
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'ls -lh /opt/models/*.gguf && /opt/llama.cpp/build/bin/llama-cli -m /opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf -n 1'

# Check Node 3 (size check only, llama-cli test may fail)
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'ls -lh /opt/models/*.gguf'
```

**Acceptable Ranges**:
- OR1-Behemoth: 75-85 GB (83 GB expected)
- Strand-Rust-Coder: 6.5-8.5 GB (7.4 GB expected)
- DeepSeek V3: 110-130 GB (120 GB expected)

### Phase 2: Model Router Configuration (10 minutes)

**Objective**: Deploy model selector script that dynamically selects model based on task type.

#### Step 2A: Create Model Router Script

**What to Do**:
1. Create `/usr/local/bin/llama-model-selector.sh` on Node 1 (head node)
2. Script accepts TASK_TYPE argument and sets environment variables:
   - `LLAMA_MODEL_PATH` - Path to selected GGUF file
   - `LLAMA_MODEL_NAME` - Model identifier
3. Test script with all three task types
4. Make script executable

**Model Routing Logic**:
```bash
TASK_TYPE="analyze"      → OR1-Behemoth (deep analysis, 32-128k context)
TASK_TYPE="idiomatic"    → Strand-Rust-Coder (idiomatic improvements, 32k context)
TASK_TYPE="refactor"     → Strand-Rust-Coder (idiomatic improvements, 32k context)
TASK_TYPE="borrowing"    → DeepSeek Coder (complex ownership, 32-128k context)
TASK_TYPE="ownership"    → DeepSeek Coder (ownership patterns, 32-128k context)
TASK_TYPE="complex"      → DeepSeek Coder (complex ownership, 32-128k context)
TASK_TYPE="fix"          → Strand-Rust-Coder (idiomatic fixes, 32k context)
TASK_TYPE="complete"     → Strand-Rust-Coder (idiomatic fixes, 32k context)
TASK_TYPE="typestate"    → Strand-Rust-Coder (idiomatic fixes, 32k context)
```

**Commands** (on Node 1 - 10.0.0.31):
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cat > /usr/local/bin/llama-model-selector.sh <<'SCRIPT'
#!/bin/bash

# Simple task classifier for multi-model routing
TASK_TYPE="${1:-analyze}"

case "$TASK_TYPE" in
  "analyze"|"idiomatic"|"refactor")
    # Use OR1-Behemoth for deep analysis, idiomatic improvements
    MODEL_PATH="/opt/models/OR1-Behemoth.Q8_0.gguf"
    MODEL_NAME="OR1-Behemoth"
    ;;
  "borrowing"|"ownership"|"complex")
    # Use DeepSeek Coder for difficult reasoning, ownership puzzles
    MODEL_PATH="/opt/models/DeepSeek-Coder-V3.Q5_K_M.gguf"
    MODEL_NAME="DeepSeek-Coder-V3"
    ;;
  "fix"|"complete"|"typestate")
    # Use Strand-Rust-Coder for idiomatic fixes
    MODEL_PATH="/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf"
    MODEL_NAME="Strand-Rust-Coder"
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

# Test script with all task types
for task_type in analyze idiomatic borrowing; do
  echo "Testing task_type: $task_type"
  /usr/local/bin/llama-model-selector.sh $task_type
done

echo "Model router script deployed and tested"
EOF
```

**Expected Time**: 10 minutes

### Phase 3: Multi-Model Service Deployment (15 minutes)

**Objective**: Create and enable systemd services for RPC workers and head node.

#### Step 3A: Deploy RPC Worker Service on Node 2

**What to Do**:
1. Create `/etc/systemd/system/llama-rpc.service` on Node 2
2. Configure environment: `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES=0`
3. Configure model: Strand-Rust-Coder-14B-v1.Q8_0.gguf (7 GB)
4. Set `Restart=always`, `RestartSec=5`
5. Set `LimitNOFILE=65535`
6. Reload systemd daemon
7. Enable service (do NOT start yet)
8. Verify service status

**Commands** (on Node 2 - 10.0.0.32 via pve2):
```bash
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-rpc.service <<'SERVICE'
[Unit]
Description=Llama.cpp RPC Worker (Strand-Rust-Coder 14B)
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

# Reload systemd daemon
systemctl daemon-reload

# Enable service (don't start yet)
systemctl enable llama-rpc

# Verify service exists
systemctl status llama-rpc
EOF
```

#### Step 3B: Deploy RPC Worker Service on Node 3

**What to Do**:
1. Create `/etc/systemd/system/llama-rpc.service` on Node 3
2. Configure environment: `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES=0`
3. Configure model: DeepSeek-Coder-V3.Q5_K_M.gguf (~120 GB)
4. Set `Restart=always`, `RestartSec=5`
5. Set `LimitNOFILE=65535`
6. Reload systemd daemon
7. Enable service (do NOT start yet)
8. Verify service status

**Commands** (on Node 3 - 10.0.0.33 via pve3):
```bash
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-rpc.service <<'SERVICE'
[Unit]
Description=Llama.cpp RPC Worker (DeepSeek Coder V3)
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

# Reload systemd daemon
systemctl daemon-reload

# Enable service (don't start yet)
systemctl enable llama-rpc

# Verify service exists
systemctl status llama-rpc
EOF
```

#### Step 3C: Deploy Head Node Service on Node 1

**What to Do**:
1. Create `/etc/systemd/system/llama-head.service` on Node 1
2. Configure environment: `LD_LIBRARY_PATH`
3. Configure ExecStart to use model router script
4. Set RPC backends: `--rpc 10.100.0.32:50052,10.100.0.33:50052` (InfiniBand IPs)
5. Set model path placeholder: `$LLAMA_MODEL_PATH` (set by router script)
6. Set launch parameters: `--ctx-size 4096`, `-ngl 999`, `--parallel 1`, `--threads 40`
7. Set `Restart=always`, `RestartSec=10`
8. Set `LimitNOFILE=65535`, `MemoryMax=100G`
9. Reload systemd daemon
10. Enable service (do NOT start yet)
11. Verify service file

**Commands** (on Node 1 - 10.0.0.31):
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
cat > /etc/systemd/system/llama-head.service <<'SERVICE'
[Unit]
Description=Llama.cpp Head Server (Multi-Model Rust Coding)
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin

# Environment variables
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"

# Launch command with model router
# NOTE: --rpc uses InfiniBand IPs (10.100.0.x) for 100Gb/s
# NOTE: --parallel 1 for baseline (increase only if GPUs idle)
ExecStart=/bin/bash -c '/usr/local/bin/llama-model-selector.sh analyze && /opt/llama.cpp/build/bin/llama-server -m $LLAMA_MODEL_PATH --rpc 10.100.0.32:50052,10.100.0.33:50052 --host 0.0.0.0 --port 8000 --ctx-size 4096 -ngl 999 --parallel 1 --threads 40'

# Restart behavior
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65535
MemoryMax=100G

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd daemon
systemctl daemon-reload

# Enable service (start manually after model download completes)
systemctl enable llama-head

# Verify service file created
cat /etc/systemd/system/llama-head.service
EOF

# Verify service file
systemctl cat llama-head
EOF
```

### Phase 4: Launch & Verification (20 minutes)

**Objective**: Start all services and verify they're working correctly.

#### Step 4A: Launch RPC Workers

**What to Do**:
1. Start RPC worker on Node 2: `systemctl start llama-rpc`
2. Start RPC worker on Node 3: `systemctl start llama-rpc`
3. Wait 5 seconds for initialization
4. Verify both services are active: `systemctl is-active llama-rpc`
5. Check listening ports: `ss -tlnp | grep 50052`
6. Check RPC server logs: `journalctl -u llama-rpc -n 20`
7. Verify CUDA GPU detected in logs

**Commands**:
```bash
# Start Node 2 RPC worker
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'systemctl start llama-rpc && systemctl status llama-rpc'

# Start Node 3 RPC worker
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'systemctl start llama-rpc && systemctl status llama-rpc'

# Verify both are running
for node in 32 33; do
  echo "=== Node 10.0.0.$node ==="
  if [ "$node" = "32" ]; then
    ssh -o ConnectTimeout=5 root@100.127.30.114 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && ss -tlnp | grep 50052'"
  else
    ssh -o ConnectTimeout=5 root@100.68.22.98 "ssh root@10.0.0.$node 'systemctl is-active llama-rpc && ss -tlnp | grep 50052'"
  fi
  echo ""
done
```

**Expected Output**:
- Both services report "active"
- Port 50052 listening on 0.0.0.0 on both nodes
- Logs show "CUDA device 0: Tesla V100S-PCIE-32GB"

#### Step 4B: Launch Head Node (After Model Download Completes)

**What to Do**:
1. Verify all 3 model files exist on respective nodes
2. Verify file sizes are correct (within 10% of expected)
3. Check available disk space on Node 1 (should have >77 GB free)
4. Start head node service: `systemctl start llama-head`
5. Wait 10 seconds for initialization
6. Check service status: `systemctl status llama-head`
7. Review startup logs: `journalctl -u llama-head -n 50`
8. Verify model loaded successfully
9. Verify RPC backends connected (check logs)
10. Verify API endpoint listening: `ss -tlnp | grep 8000`

**Commands** (on Node 1 - 10.0.0.31):
```bash
# Verify all models exist and sizes are correct
echo "=== Model Files Check ==="
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
# Check Node 1 model
ls -lh /opt/models/OR1-Behemoth.Q8_0.gguf

# Check Node 2 model via remote
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 "ls -lh /opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf"

# Check Node 3 model size via remote (llama-cli test may fail due to size)
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 "ls -lh /opt/models/DeepSeek-Coder-V3.Q5_K_M.gguf"

# Check disk space
df -h /
EOF

# Start head service when all models verified
echo "=== Starting Head Node ==="
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
MODEL_SIZE=$(stat -f%z /opt/models/OR1-Behemoth.Q8_0.gguf 2>/dev/null || stat -c%s /opt/models/OR1-Behemoth.Q8_0.gguf)
EXPECTED_SIZE=83002634240

if [ "$MODEL_SIZE" -ge 80000000000 ]; then
  echo "Model download complete (${MODEL_SIZE} bytes)"
  systemctl start llama-head
  echo "Head server starting..."
  sleep 10
  systemctl status llama-head
else
  echo "Model download incomplete or missing (${MODEL_SIZE} bytes)"
  exit 1
fi
EOF
```

#### Step 4C: Basic API Health Check

**What to Do**:
1. Wait 15 seconds for head node to fully initialize
2. Check logs for any errors: `journalctl -u llama-head -n 50`
3. Test API endpoint with simple request
4. Verify response format (OpenAI-compatible)
5. Verify response time < 5 seconds
6. Check model identifier in response

**Commands** (from Node 1 or local machine):
```bash
# Wait for initialization
sleep 15

# Test API endpoint
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Say \"Hello from agentic Rust cluster!\""}],
    "max_tokens": 50
  }' \
  -w "\nResponse time: %{time_total}s\n" \
  -s > /tmp/api_response.json && cat /tmp/api_response.json
```

**Expected Output**:
```json
{
  "model": "OR1-Behemoth",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello from agentic Rust cluster!"
    },
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 5,
    "total_tokens": 13
  },
  "created": 1710715391
}
Response time: ~1-2 seconds
```

**Success Criteria**:
- ✅ HTTP 200 response
- ✅ Valid JSON response
- ✅ Response contains expected message
- ✅ Response time < 5 seconds
- ✅ Model identifier correct

---

## Critical Constraints & Requirements

### Hardware

- **V100S Limitation**: Lacks INT4 tensor cores → Q8_0 better than Q4_K_M
  - Q8_0: ~15-20 tokens/s on V100S (near memory bandwidth)
  - Q4_K_M: ~8-12 tokens/s on V100S (de-quantization bottleneck)
- **Model Sizes**:
  - OR1-Behemoth Q8_0: 77 GB (fits in 32 GB with 6 GB KV cache)
  - Strand-Rust-Coder Q8_0: 7 GB (fits comfortably)
  - DeepSeek Coder V3 Q5: 120 GB (tight fit, monitor closely)

### Network

- **MUST Use InfiniBand IPs** for RPC communication (10.100.0.x)
  - Node 2: 10.100.0.32:50052
  - Node 3: 10.100.0.33:50052
  - Do NOT use Ethernet IPs (10.0.0.x) for RPC backends
- **Expected Bandwidth**: >50 Gbps sustained during inference
- **MTU**: 2044 (IPoIB) / 4096 (Hardware)

### Software

- **llama.cpp**: v7760 built with GCC 12.2.1, CUDA 12.2
- **Systemd**: Auto-restart enabled for all services
- **Model Router**: Script-based routing on head node

### Beads Tracking

**Epic ID**: `beefcake2-lhr0`
**View Epic**: `bd show beefcake2-lhr0`
**View Ready Tasks**: `bd ready`
**Update Progress**: `bd update <task-id> --status in_progress --notes "progress notes"`
**Complete Task**: `bd close <task-id> --reason "completed successfully"`

---

## Troubleshooting Guide

### Model Download Issues

**Issue**: Download slow or failing
**Solution**:
```bash
# Resume with -c flag
cd /opt/models
wget -c "URL" -O file.gguf

# Check download progress
ps aux | grep wget
ls -lh file.gguf
```

**Issue**: Disk space full
**Solution**:
```bash
# Stop download and delete partial files
pkill wget
rm -f /opt/models/*.gguf.part*
df -h /

# Resume with smaller model or clear space
```

### RPC Worker Issues

**Issue**: Service won't start
**Solution**:
```bash
# Check logs
journalctl -u llama-rpc -n 100

# Check GPU access
nvidia-smi

# Restart service
systemctl restart llama-rpc
```

**Issue**: Port 50052 not listening
**Solution**:
```bash
# Check port
ss -tlnp | grep 50052

# Check firewall
firewall-cmd --list-ports

# Restart service
systemctl restart llama-rpc
```

### Head Node Issues

**Issue**: Service won't start
**Solution**:
```bash
# Check logs
journalctl -u llama-head -n 100

# Check model file exists
ls -lh /opt/models/OR1-Behemoth.Q8_0.gguf

# Check model router script
ssh root@10.0.0.31 "/usr/local/bin/llama-model-selector.sh analyze"

# Restart service
systemctl restart llama-head
```

**Issue**: API not responding
**Solution**:
```bash
# Check service status
systemctl status llama-head

# Check port
ss -tlnp | grep 8000

# Check logs
journalctl -u llama-head -f

# Restart service
systemctl restart llama-head
```

**Issue**: RPC backends not connected
**Solution**:
```bash
# Check RPC workers are running
for node in 32 33; do
  echo "=== Node $node ==="
  ssh root@10.0.0.$node "systemctl is-active llama-rpc"
done

# Check connectivity
ping -c 3 10.100.0.32
ping -c 3 10.100.0.33

# Check RPC server logs
ssh root@10.0.0.32 "journalctl -u llama-rpc -n 50"
ssh root@10.0.0.33 "journalctl -u llama-rpc -n 50"
```

### GPU Utilization Issues

**Issue**: Only 1 GPU active during inference
**Possible Causes**:
1. RPC workers not running
2. Incorrect RPC backend IPs (using Ethernet instead of InfiniBand)
3. Model not distributing layers properly
4. Network bottleneck

**Solution**:
```bash
# Check all 3 GPUs
for node in 31 32 33; do
  echo "=== Node 10.0.0.$node ==="
  case $node in
    31) ssh -o ConnectTimeout=5 -J root@100.127.208.104 root@10.0.0.$node "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits" ;;
    32) ssh -o ConnectTimeout=5 root@100.127.30.114 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits\"" ;;
    33) ssh -o ConnectTimeout=5 root@100.68.22.98 "ssh root@10.0.0.$node \"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits\"" ;;
  esac
done

# Check RPC connectivity
ping -c 3 10.100.0.32
ping -c 3 10.100.0.33
```

**Expected Behavior**: All 3 GPUs should show >50% utilization during inference

---

## Performance Targets

### Per Model

| Model | Target Throughput | Acceptable | Critical |
|--------|------------------|------------|----------|
| **OR1-Behemoth 73B** | 15-20 tokens/s | >10 tokens/s | <8 tokens/s |
| **Strand-Rust-Coder 14B** | 20-30 tokens/s | >15 tokens/s | <10 tokens/s |
| **DeepSeek Coder V3 671B** | 10-15 tokens/s | >8 tokens/s | <5 tokens/s |

### Overall Cluster

| Metric | Target | Acceptable | Critical |
|--------|---------|------------|----------|
| **Average Throughput** | 15-25 tokens/s | >10 tokens/s | <8 tokens/s |
| **Time to First Token** | <2 seconds | <5 seconds | >10 seconds |
| **GPU Utilization** | 70-90% | >50% | <40% |
| **Memory/GPU** | 16-26 GB | <30 GB | >30 GB |
| **Network (IB)** | >50 Gbps | >30 Gbps | <20 Gbps |

---

## Documentation References

**All documentation is consolidated in**: `/Users/briansquires/beefcake2/docs/agentic-rust-cluster/`

### Key Files

1. **`README.md`** - Comprehensive overview (START HERE)
2. **`distributed-llama-production-guide.md`** - Complete deployment guide
3. **`deployment-strategy-update.md`** - Strategy comparison
4. **`hybrid-model-strategy.md`** - Model analysis
5. **`beads-epic-summary.md`** - Beads epic (17 tasks, 6 phases)
6. **`CONSOLIDATION-SUMMARY.md`** - Consolidation summary
7. **`NEXT-AGENT-PROMPT.md`** - This prompt

### Quick Navigation

**Start Here** → `docs/agentic-rust-cluster/README.md`
**Strategy** → `docs/agentic-rust-cluster/deployment-strategy-update.md`
**Deployment** → `docs/agentic-rust-cluster/distributed-llama-production-guide.md`
**Model Analysis** → `docs/agentic-rust-cluster/hybrid-model-strategy.md`
**Beads Tasks** → `docs/agentic-rust-cluster/beads-epic-summary.md`

---

## Success Criteria for Phase 1

### Multi-Model Acquisition Complete When:

- ✅ All 3 model files downloaded successfully
- ✅ File sizes match expected values (±10% tolerance)
- ✅ Model integrity verified with `llama-cli`
- ✅ Disk space sufficient on all nodes
- ✅ Download time within estimated range (65-80 minutes total)
- ✅ All model files accessible on respective nodes

### Phase 1 Complete When:

- ✅ OR1-Behemoth 73B Q8_0 downloaded on Node 1 (77 GB)
- ✅ Strand-Rust-Coder 14B Q8_0 downloaded on Node 2 (7 GB)
- ✅ DeepSeek Coder V3 671B Q5_K_M downloaded on Node 3 (~120 GB)
- ✅ All models verified and ready
- ✅ Disk space adequate for subsequent phases

**Then Proceed to Phase 2** (Model Router Configuration)

---

## Commands Summary

### Quick Status Check

```bash
# Check Beads epic status
cd /Users/briansquires/beefcake2
bd show beefcake2-lhr0

# Check ready tasks
bd ready

# Check cluster status (services not started yet)
for node in 31 32 33; do
  echo "=== Node 10.0.0.$node ==="
  case $node in
    31) ssh -o ConnectTimeout=5 -J root@100.127.208.104 root@10.0.0.$node "systemctl status llama-* 2>/dev/null || echo 'no services'" ;;
    32) ssh -o ConnectTimeout=5 root@100.127.30.114 "ssh root@10.0.0.$node 'systemctl status llama-* 2>/dev/null || echo 'no services'" ;;
    33) ssh -o ConnectTimeout=5 root@100.68.22.98 "ssh root@10.0.0.$node 'systemctl status llama-* 2>/dev/null || echo 'no services'" ;;
  esac
  echo ""
done
```

---

## Important Notes

1. **SSH Access Patterns**:
   - Use Proxmox hosts as jump servers for direct VM access
   - Node 1: `ssh root@100.127.208.104 root@10.0.0.31`
   - Node 2: `ssh root@100.127.30.114 root@10.0.0.32`
   - Node 3: `ssh root@100.68.22.98 root@10.0.0.33`

2. **InfiniBand IPs** (Critical):
   - RPC backends MUST use 10.100.0.x (NOT 10.0.0.x)
   - Network: 100 Gb/s HDR100 InfiniBand
   - MTU: 2044 (IPoIB) / 4096 (Hardware)

3. **Model Router Script**:
   - Automatically selects model based on TASK_TYPE
   - Exports `LLAMA_MODEL_PATH` and `LLAMA_MODEL_NAME`
   - Default: OR1-Behemoth (deep analysis)
   - Test with: `/usr/local/bin/llama-model-selector.sh analyze`

4. **Systemd Services**:
   - Auto-restart enabled (Restart=always)
   - Survive SSH disconnects
   - Logs: `journalctl -u <service_name> -f`

5. **Beads Tracking**:
   - Update task status: `bd update <task-id> --status in_progress --notes "notes"`
   - Complete task: `bd close <task-id> --reason "completed successfully"`
   - View ready: `bd ready`

---

**Last Updated**: January 16, 2026
**Status**: 🚀 Ready to Begin Phase 1 - Multi-Model Acquisition

**Estimated Total Time**: ~2.5-3.5 hours for complete deployment (Phases 1-4)
