# Agentic Rust Coding Cluster Documentation

**Project**: Distributed inference cluster for agentic Rust coding using llama.cpp with RPC
**Status**: 📋 Hybrid Model Strategy (Multi-Model Deployment)
**Last Updated**: January 16, 2026

## Quick Start

**New to this project?** Start here: [deployment-strategy-update.md](deployment-strategy-update.md)

**Ready to deploy?** See [distributed-llama-production-guide.md](distributed-llama-production-guide.md)

## Document Directory

```
docs/agentic-rust-cluster/
├── README.md (this file)
├── distributed-llama-production-guide.md
├── deployment-strategy-update.md
├── hybrid-model-strategy.md
├── beads-epic-summary.md
└── deployment-summary.md
```

## Document Overview

| File | Purpose | When to Read |
|-------|-----------|--------------|
| **README.md** | This file - overview and quick navigation | **Start here** |
| **distributed-llama-production-guide.md** | Complete production deployment guide with systemd services, model router script, and all commands | **Deploying** |
| **deployment-strategy-update.md** | Updated deployment plan comparing original vs hybrid strategy with model selection analysis | **Planning** |
| **hybrid-model-strategy.md** | Deep analysis of 2026 agentic Rust models (OR1-Behemoth, Strand-Rust-Coder, DeepSeek Coder) | **Understanding** |
| **beads-epic-summary.md** | Complete beads epic structure with 17 tasks across 6 phases | **Tracking** |
| **deployment-summary.md** | Quick reference summary with 6-step deployment plan | **Reference** |

## Project Overview

### Objective

Deploy a production-grade distributed inference cluster for agentic Rust coding tasks using llama.cpp with RPC backend. The cluster uses a **hybrid multi-model strategy** optimized for different task types.

### Infrastructure

- **3× Proxmox VMs** with Tesla V100S 32GB GPUs
- **InfiniBand ConnectX-6 HDR100** @ 100 Gb/s for RPC communication
- **Systemd services** for production reliability (auto-restart)
- **Model router** for dynamic model selection based on task type

### Hardware

| Node | Role | GPU | VRAM | IP (Eth) | IP (IB) |
|-------|-------|------|-------|-----------|---------|
| **Node 1** | Head | V100S 32GB | 10.0.0.31 | 10.100.0.31 |
| **Node 2** | RPC Worker | V100S 32GB | 10.0.0.32 | 10.100.0.32 |
| **Node 3** | RPC Worker | V100S 32GB | 10.0.0.33 | 10.100.0.33 |

### Models Deployed

| Model | Parameters | Size | VRAM (Q8) | Node | Use Case | Strengths | Weaknesses |
|--------|-----------|-------|--------------|-------|----------|-----------|-----------|
| **OR1-Behemoth 73B** | 73B (embiggened) | 77 GB | 1 | Deep analysis, large context | Large context (32-128k), detailed explanations | Coherence drift, stability issues |
| **Strand-Rust-Coder 14B** | 14B (swarm) | 7 GB | 2 | Idiomatic analysis, real-time suggestions | 94.3% compile rate, peer-reviewed | Limited context (32k), fast iteration |
| **DeepSeek Coder V3 671B** | 671B (MoE) | 120 GB | 3 | Complex ownership, novel patterns | Self-correction, massive knowledge base | Massive model, slow (MoE latency), resource-intensive |

### Model Routing Strategy

Task-based routing via `/usr/local/bin/llama-model-selector.sh` on head node:

```bash
# Task types and assigned models
TASK_TYPE="analyze"    → OR1-Behemoth 73B (deep analysis)
TASK_TYPE="idiomatic"  → Strand-Rust-Coder 14B (idiomatic improvements)
TASK_TYPE="refactor"     → Strand-Rust-Coder 14B (idiomatic improvements)
TASK_TYPE="borrowing"    → DeepSeek Coder V3 671B (complex ownership)
TASK_TYPE="ownership"    → DeepSeek Coder V3 671B (ownership patterns)
TASK_TYPE="complex"      → DeepSeek Coder V3 671B (complex ownership)
TASK_TYPE="fix"         → Strand-Rust-Coder 14B (idiomatic fixes)
TASK_TYPE="complete"     → Strand-Rust-Coder 14B (idiomatic fixes)
TASK_TYPE="typestate"    → Strand-Rust-Coder 14B (idiomatic fixes)
```

## Deployment Phases

### Phase 1: Multi-Model Acquisition (3 tasks)

**Estimated Time**: 65-80 minutes (includes all model downloads)

| Task | Model | Size | Target Node | Time |
|-------|--------|-------|------------|--------|
| Download OR1-Behemoth 73B Q8_0 | 77 GB | Node 1 | 20-25 min |
| Download Strand-Rust-Coder 14B Q8_0 | 7 GB | Node 2 | 5 min |
| Download DeepSeek Coder V3 671B Q5_K_M | 120 GB | Node 3 | 40-50 min |

**Status**: ⏳ Not Started

### Phase 2: Model Router Configuration (1 task)

**Estimated Time**: 10 minutes

| Task | Description | Dependencies |
|-------|-------------|--------------|
| Deploy model selector script | Create `/usr/local/bin/llama-model-selector.sh` on Node 1 | None |

**Status**: ⏳ Not Started

### Phase 3: Multi-Model Service Deployment (3 tasks)

**Estimated Time**: 15 minutes

| Task | Node | Model | Service | Time |
|-------|------|--------|---------|--------|
| Deploy RPC worker (Strand) | Node 2 | llama-rpc (Strand 14B) | 5 min |
| Deploy RPC worker (DeepSeek) | Node 3 | llama-rpc (DeepSeek V3) | 5 min |
| Deploy updated multi-model head service | Node 1 | llama-head (multi-model router) | 5 min |

**Status**: ⏳ Not Started

### Phase 4: Launch & Verification (4 tasks)

**Estimated Time**: 20 minutes

| Task | Description | Dependencies |
|-------|-------------|--------------|
| Launch all RPC workers | Start both llama-rpc services | Phase 3 tasks |
| Launch head node with model router | Start llama-head service | Phase 1-3 tasks |
| Test API with all 3 models | Verify each model responds correctly | Launch tasks |
| Verify distributed GPU usage | Monitor all 3 GPUs during inference | Launch + API test tasks |

**Status**: ⏳ Not Started

### Phase 5: Multi-Model Performance Testing (6 tasks)

**Estimated Time**: 2-3 hours

| Task | Model | Test Focus | Time |
|-------|--------|-------------|--------|
| Baseline OR1-Behemoth | Deep analysis, architectural refactoring | 30 min |
| Baseline Strand-Rust-Coder | Idiomatic patterns, real-time suggestions | 30 min |
| Baseline DeepSeek Coder | Complex ownership, borrow checker puzzles | 30 min |
| Compare throughput across models | Identify optimal model per task type | 30 min |
| Identify optimal model per task type | Create task-to-model mapping | 30 min |
| Document performance characteristics | Throughput, latency, GPU utilization | 30 min |

**Status**: ⏳ Not Started

## Key Configuration Details

### Head Node Service (llama-head)

**Location**: `/etc/systemd/system/llama-head.service` (Node 1)

**Key Configuration**:
```ini
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"
ExecStart=/bin/bash -c '/usr/local/bin/llama-model-selector.sh analyze && /opt/llama.cpp/build/bin/llama-server -m $LLAMA_MODEL_PATH --rpc 10.100.0.32:50052,10.100.0.33:50052 --host 0.0.0.0 --port 8000 --ctx-size 4096 -ngl 999 --parallel 1 --threads 40'
```

**Environment Variables**:
- `LLAMA_MODEL_PATH` - Set by model selector script
- `LLAMA_MODEL_NAME` - Set by model selector script
- `TASK_TYPE` - Set by model selector script

### RPC Worker Services

**Node 2 (Strand-Rust-Coder 14B)**:
- **Service**: `llama-rpc.service`
- **Model**: `/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf` (7 GB)
- **Port**: 50052

**Node 3 (DeepSeek Coder V3 671B)**:
- **Service**: `llama-rpc.service`
- **Model**: `/opt/models/DeepSeek-Coder-V3-Q5_K_M.gguf` (120 GB)
- **Port**: 50052

### Model Selector Script

**Location**: `/usr/local/bin/llama-model-selector.sh` (Node 1)

**Purpose**: Dynamically select model based on task type

**Usage**:
```bash
# Set task type (usually by API request)
export TASK_TYPE="analyze"  # or idiomatic, refactor, borrowing, etc.

# Run through model router
/usr/local/bin/llama-model-selector.sh $TASK_TYPE
```

**Supported Task Types**:
- `analyze` - Deep analysis, large context tasks → OR1-Behemoth 73B
- `idiomatic` - Idiomatic improvement, real-time suggestions → Strand-Rust-Coder 14B
- `refactor` - Code refactoring, architectural improvements → Strand-Rust-Coder 14B
- `borrowing` - Complex ownership, lifetime issues → DeepSeek Coder V3 671B
- `ownership` - Ownership pattern analysis → DeepSeek Coder V3 671B
- `complex` - Complex cross-module logic → DeepSeek Coder V3 671B
- `fix`, `complete`, `typestate` - Various improvements → Strand-Rust-Coder 14B

## Performance Targets

### Per Model

| Model | Target Throughput | Expected Latency | Compilation Rate | Context |
|--------|------------------|-------------------|----------------|----------|
| **OR1-Behemoth 73B** | 15-20 tokens/s | 2-3 seconds | ~70-90% | 32-128k |
| **Strand-Rust-Coder 14B** | 20-30 tokens/s | <1 second | 94.3% | 32k |
| **DeepSeek Coder V3 671B** | 10-15 tokens/s | 3-5 seconds (MoE) | ~80-85% | 32-128k |

### Overall Cluster Targets

| Metric | Target | Acceptable | Critical |
|--------|---------|------------|----------|
| Throughput (avg across models) | 12-25 tokens/s | >10 tokens/s | <8 tokens/s |
| Time to First Token | <3 seconds | <5 seconds | >10 seconds |
| GPU Utilization | 60-90% | >50% | <40% |
| Memory/GPU (varies by model) | 16-26 GB | <30 GB | >30 GB |
| Network (InfiniBand) | >50 Gbps | >30 Gbps | <20 Gbps |

## Beads Issue Tracking

**Epic ID**: `beefcake2-lhr0`
**Epic Title**: Distributed OR1-Behemoth 72B Inference Cluster - Production Deployment (Updated with Hybrid Strategy)
**Priority**: P0 (Highest)

**View Epic**:
```bash
cd /Users/briansquires/beefcake2
bd show beefcake2-lhr0
```

**View Tasks**:
```bash
cd /Users/briansquires/beefcake2
bd ready
```

**Progress Summary**:
- Total tasks: 17
- Status: All ready (⏳ Not Started)
- Next phase: Phase 1 (Multi-Model Acquisition)

## Quick Reference Commands

### Check Cluster Status
```bash
# Check all services
for node in 31 32 33; do
  echo "=== Node 10.0.0.$node ==="
  ssh root@10.0.0.$node "systemctl status llama-*"
done

# Check GPU status
for node in 31 32 33; do
  echo "=== Node 10.0.0.$node ==="
  ssh root@10.0.0.$node "nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader,nounits"
done

# Check model selector
ssh root@10.0.0.31 "/usr/local/bin/llama-model-selector.sh analyze"
```

### Test All Models
```bash
# Test OR1-Behemoth (deep analysis)
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Analyze this crate for idiomatic patterns"}],
    "max_tokens": 500
  }'

# Test Strand-Rust-Coder (idiomatic)
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Strand-Rust-Coder",
    "messages": [{"role": "user", "content": "Suggest idiomatic improvements for this function"}],
    "max_tokens": 200
  }'

# Test DeepSeek Coder (complex ownership)
curl -X POST "http://10.0.0.31:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-Coder-V3",
    "messages": [{"role": "user", "content": "Solve this lifetime error"}],
    "max_tokens": 300
  }'
```

### Monitor Distributed GPUs
```bash
# Monitor all 3 GPUs
watch -n 1 'echo "=== Distributed GPU Utilization ==="
for node in 31 32 33; do
  echo "Node 10.0.0.$node:"
  ssh root@10.0.0.$node "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits"
  echo ""
done'
```

### Service Management
```bash
# Start all services
systemctl start llama-rpc  # Run on Node 2 & 3
systemctl start llama-head # Run on Node 1

# Stop all services
systemctl stop llama-head
systemctl stop llama-rpc

# Restart services
systemctl restart llama-head
systemctl restart llama-rpc

# View logs
journalctl -u llama-head -f
journalctl -u llama-rpc -f
```

## Troubleshooting

### Model Not Loading

```bash
# Check model files
ls -lh /opt/models/*.gguf

# Check model selector script
ssh root@10.0.0.31 "cat /usr/local/bin/llama-model-selector.sh"

# Test model with llama-cli
ssh root@10.0.0.31 "/opt/llama.cpp/build/bin/llama-cli -m /opt/models/OR1-Behemoth.Q8_0.gguf -n 1"
```

### RPC Worker Not Running

```bash
# Check service status
ssh root@10.0.0.32 "systemctl status llama-rpc"
ssh root@10.0.0.33 "systemctl status llama-rpc"

# Check logs
ssh root@10.0.0.32 "journalctl -u llama-rpc -n 50"
ssh root@10.0.0.33 "journalctl -u llama-rpc -n 50"

# Restart if needed
ssh root@10.0.0.32 "systemctl restart llama-rpc"
ssh root@10.0.0.33 "systemctl restart llama-rpc"
```

### API Not Responding

```bash
# Check head service
ssh root@10.0.0.31 "systemctl status llama-head"

# Check logs
ssh root@10.0.0.31 "journalctl -u llama-head -n 50"

# Check port
ssh root@10.0.0.31 "ss -tlnp | grep 8000"

# Test endpoint
curl -v http://10.0.0.31:8000/v1/models
```

### GPU Not Utilized

```bash
# Check GPU status
for node in 31 32 33; do
  echo "=== Node 10.0.0.$node ==="
  ssh root@10.0.0.$node "nvidia-smi"
done

# Check if model is offloading to GPU
ssh root@10.0.0.31 "journalctl -u llama-head | grep -i gpu\|layer\|ngl"
```

## Background & Analysis

### Why This Strategy?

The original plan used OR1-Behemoth 73B on all 3 nodes. Based on comprehensive analysis of 2026 agentic Rust models, we switched to a hybrid strategy:

**OR1-Behemoth Issues**:
- "Embiggened" architecture (73B scaled from 32B)
- Coherence drift and repetition loops
- Stability concerns
- Massive hardware requirements (77GB VRAM per node)

**Strand-Rust-Coder 14B Advantages**:
- 94.3% compilation rate (peer-reviewed swarm training)
- +13% improvement over larger models
- Idiomatic Rust specialization
- Fits in 7GB VRAM (consumer GPU friendly)
- Fast latency for real-time suggestions

**DeepSeek Coder V3 671B Capabilities**:
- Self-correction mechanism (critical for complex borrow checker issues)
- Massive knowledge base (System 2 reasoning)
- Unmatched at solving novel ownership patterns
- 80-85% compilation rate

### Key Insights

1. **Task Optimization**: Match model capability to task type
   - OR1 for deep analysis, large context
   - Strand for idiomatic analysis, real-time suggestions
   - DeepSeek for complex ownership puzzles

2. **Hardware Efficiency**: Hybrid uses less total VRAM
   - Original: 77GB × 3 = 231GB (doesn't fit)
   - Revised: 77GB + 7GB + 120GB = 204GB (fits)

3. **Risk Mitigation**: Multiple models reduce single-point failure
   - If one model hallucinates, alternatives available
   - Peer-reviewed Strand has higher reliability than embiggened OR1

4. **Benchmark Data**: Strand proven Rust performance
   - RustEvo² benchmark: 94.3% compile rate
   - Peer-reviewed training ensures idiomatic safety

5. **Flexibility**: Easy to add new models via router script
   - No infrastructure changes required
   - Script-based routing is maintainable

## Performance Metrics to Collect

For each model, measure:

1. **Throughput** (tokens/second)
2. **Time to First Token** (seconds)
3. **Compilation Rate** (from test code)
4. **Idiomatic Adherence** (from code review)
5. **GPU Utilization** (percentage during inference)
6. **Memory Usage** (GB per GPU)
7. **Latency** (per token)

## Next Steps for New Agent

1. **Start with Phase 1** - Download all three models
2. **Deploy Phase 2** - Model router configuration
3. **Deploy Phase 3** - All three systemd services
4. **Verify Phase 4** - Launch and test all models
5. **Benchmark Phase 5** - Collect performance data
6. **Document results** - Update this README with findings

## File Index

See [beads-epic-summary.md](beads-epic-summary.md) for complete task breakdown.

## Related Documentation

- [hybrid-model-strategy.md](hybrid-model-strategy.md) - Deep analysis of 2026 agentic models
- [deployment-strategy-update.md](deployment-strategy-update.md) - Strategy comparison and justification
- [distributed-llama-production-guide.md](distributed-llama-production-guide.md) - Complete deployment guide
- [deployment-summary.md](deployment-summary.md) - Quick reference summary

## Architecture Diagram

```
                    ┌─────────────────┐
                    │   Node 1       │
                    │  (Head)        │
                    │                 │
                    │  llama-head     │
                    │  (Multi-Model   │
                    │   Router)       │
                    └─────┬──────────┘
                          │
             ┌────────────┴────────────┐
             │                          │
    ┌────────┴────────┐   ┌────────┴────────┐
    │                │   │                │
    │   Node 2       │   │   Node 3       │
    │  (RPC Worker)  │   │  (RPC Worker)  │
    │                │   │                │
    │ llama-rpc      │   │  llama-rpc      │
    │ (Strand 14B)   │   │ (DeepSeek V3)   │
    │                │   │                │
    └────────────────┘   └────────────────┘

InfiniBand Network (10.100.0.x @ 100 Gb/s)
```

---

**Last Updated**: January 16, 2026
**Maintainer**: TheFermiSea
**Status**: 📋 Ready for Deployment - Phase 1: Multi-Model Acquisition
