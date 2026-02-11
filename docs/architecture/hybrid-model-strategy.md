# Updated Deployment Strategy - Hybrid Model Approach

**Date**: January 16, 2026
**Status**: Plan Updated Based on Agentic Rust Model Analysis

## Overview

After comprehensive analysis of 2026 agentic Rust coding models, we've revised the deployment strategy from single-model (OR1-Behemoth 73B) to **hybrid multi-model approach** optimized for different task types.

## Key Insights from Analysis

### OR1-Behemoth 73B Issues

**Strengths**:
- 73B parameters provide large context window (32-128k tokens)
- Trained on Tesslate/Rust_Dataset (idiomatic patterns, no_std)
- Good for explaining code and providing detailed examples

**Weaknesses** (for our use case):
- "Embiggened" architecture (Qwen3-72B scaled from 32B)
- Coherence drift and repetition loops reported by users
- Stability concerns without extensive continued pre-training
- **Massive hardware requirements**: 145GB VRAM (FP16), 77GB (Q8)
- Less suited for "deep analysis" or complex architectural refactoring

### Strand-Rust-Coder 14B-v1 Advantages

**Strengths** (for our use case):
- **14B parameters** (fits easily in 12GB VRAM at Q8)
- **94.3% compilation rate** (peer-reviewed swarm synthesis)
- **+13% improvement** on RustEvo² benchmark vs larger models
- Specialized for **Typestate patterns, generics, no_std code**
- **Fast latency** (~10 tokens/sec iteration for real-time analysis)
- High adherence to idiomatic Rust (peer review filtering)
- Runs on consumer GPUs (RTX 4090, MacBook Pro) or single V100

**Weaknesses**:
- Limited context window (32k tokens)
- Smaller knowledge base than OR1/DeepSeek
- Less suited for complex cross-module reasoning

### DeepSeek Coder V3 671B Capabilities

**Strengths** (for specific tasks):
- **671B Mixture-of-Experts** with ~37B active parameters
- **Self-correction** (critical for complex borrow checker issues)
- **System 2 reasoning** (massive knowledge base)
- **Unmatched at solving novel ownership patterns** (the "borrow checker puzzle")
- 80-85% compilation rate (better than most generalist models)

**Weaknesses**:
- **Massive model** - requires significant VRAM even at Q5 quantization
- **MoE routing latency** (can be slow for single-prompt tasks)
- Resource-intensive for typical deployments

## Revised Deployment Architecture

### Node Assignment Strategy

| Node | Hardware | Primary Model | Secondary Models | Use Case |
|-------|-----------|---------------|-----------------|-----------|
| **Node 1** (Head) | V100S 32GB | OR1-Behemoth 73B Q8 | Strand 14B Q8 (Node2), DeepSeek V3 Q5 (Node3) | Deep analysis, complex refactoring |
| **Node 2** (RPC) | V100S 32GB | Strand-Rust-Coder 14B Q8 | - | Idiomatic analysis, real-time suggestions |
| **Node 3** (RPC) | V100S 32GB | DeepSeek Coder V3 671B Q5 | - | Complex ownership issues, novel patterns |

**Total VRAM Requirements**:
- Node 1: 77GB (OR1 primary) + 12GB (Strand alternative)
- Node 2: 7GB (Strand Q8)
- Node 3: ~120GB (DeepSeek Q5 - estimated)
- **Fits comfortably** in 3×32GB cluster (96GB total)

### Model Routing Strategy

**Script-based routing** on head node determines model based on task type:

```bash
TASK_TYPE="analyze" → OR1-Behemoth 73B Q8
TASK_TYPE="idiomatic" → Strand-Rust-Coder 14B Q8
TASK_TYPE="refactor" → Strand-Rust-Coder 14B Q8
TASK_TYPE="borrowing" → DeepSeek Coder V3 671B Q5
TASK_TYPE="ownership" → DeepSeek Coder V3 671B Q5
TASK_TYPE="complex" → DeepSeek Coder V3 671B Q5
TASK_TYPE="fix" → Strand-Rust-Coder 14B Q8
TASK_TYPE="complete" → Strand-Rust-Coder 14B Q8
TASK_TYPE="typestate" → Strand-Rust-Coder 14B Q8
```

## Revised Deployment Phases

### Phase 1: Multi-Model Acquisition (3 tasks)

| Task | Model | Size | Target Node | Download Time |
|-------|--------|-------|------------|----------------|
| **1A** | OR1-Behemoth 73B Q8 | Node 1 | ~20-25 min |
| **1B** | Strand-Rust-Coder 14B Q8 | Node 2 | ~5 min |
| **1C** | DeepSeek Coder V3 Q5 | Node 3 | ~30-40 min |

**Total Download Time**: ~55-70 minutes for all models

### Phase 2: Model Router Configuration (1 task)

| Task | Description |
|-------|-------------|
| **2A** | Deploy model selector script `/usr/local/bin/llama-model-selector.sh` on head node |

### Phase 3: Systemd Service Deployment (3 tasks)

| Task | Node | Service | Model |
|-------|------|---------|--------|
| **3A** | Node 2 | llama-rpc (Strand 14B Q8) |
| **3B** | Node 3 | llama-rpc (DeepSeek V3 Q5) |
| **3C** | Node 1 | llama-head (multi-model router) |

### Phase 4: Launch & Verification (3 tasks)

| Task | Description |
|-------|-------------|
| **4A** | Launch RPC workers (Node 2: Strand, Node 3: DeepSeek) |
| **4B** | Launch head node with model routing enabled |
| **4C** | Test all three models independently and verify distributed GPU usage |

### Phase 5: Performance Baseline Testing (3 tasks)

| Task | Model | Test Focus |
|-------|--------|------------|
| **5A** | OR1-Behemoth 73B | Deep analysis, architectural refactoring |
| **5B** | Strand-Rust-Coder 14B | Idiomatic patterns, real-time suggestions |
| **5C** | DeepSeek Coder V3 | Complex ownership issues, novel patterns |

## Configuration Updates

### Head Node Service (Modified)

**Key Changes**:
- Dynamic model selection via `/usr/local/bin/llama-model-selector.sh`
- ExecStart uses script wrapper instead of direct model path
- Single service handles all models

**Updated Service File**:
```bash
[Unit]
Description=Llama.cpp Head Server (Rust Coding Models)
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64"
ExecStart=/bin/bash -c '/usr/local/bin/llama-model-selector.sh analyze && /opt/llama.cpp/build/bin/llama-server -m \$LLAMA_MODEL_PATH --rpc 10.100.0.32:50052,10.100.0.33:50052 --host 0.0.0.0 --port 8000 --ctx-size 4096 -ngl 999 --parallel 1 --threads 40'
Restart=always
RestartSec=10
LimitNOFILE=65535
MemoryMax=100G
[Install]
WantedBy=multi-user.target
```

### RPC Worker Services (Different Models)

**Node 2 Service** (Strand-Rust-Coder 14B Q8):
```bash
[Unit]
Description=Llama.cpp RPC Worker (Strand-Rust-Coder 14B)
[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64" "CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052
Restart=always
RestartSec=5
LimitNOFILE=65535
[Install]
WantedBy=multi-user.target
```

**Node 3 Service** (DeepSeek Coder V3 671B Q5):
```bash
[Unit]
Description=Llama.cpp RPC Worker (DeepSeek-Coder V3)
[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp/build/bin
Environment="LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/usr/local/cuda-12.2/lib64" "CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/llama.cpp/build/bin/rpc-server --host 0.0.0.0 --port 50052
Restart=always
RestartSec=5
LimitNOFILE=65535
[Install]
WantedBy=multi-user.target
```

## Benefits of Hybrid Strategy

### 1. Hardware Efficiency
- Strand (14B) fits comfortably in 12GB VRAM → can run on consumer GPUs
- No need for massive 77GB model for all nodes
- Lower total cluster cost

### 2. Task Optimization
- Match model capability to task type
- OR1 for deep analysis, Strand for idiomatic analysis
- DeepSeek for complex ownership puzzles
- Better performance across all use cases

### 3. Risk Mitigation
- Multiple models reduce single-point failure mode
- If one model hallucinates, alternatives available
- Peer-reviewed Strand has higher reliability than embiggened OR1

### 4. Flexibility
- Easy to add new models to routing strategy
- No infrastructure changes required
- Script-based routing is simple to maintain

## Updated Beads Tasks

Based on this analysis, the beads epic should be updated with these new tasks:

### Phase 1: Multi-Model Acquisition
- Download OR1-Behemoth 73B Q8 (primary, for deep analysis)
- Download Strand-Rust-Coder 14B Q8 (for idiomatic analysis)
- Download DeepSeek Coder V3 Q5 (for complex ownership issues)

### Phase 2: Model Router Setup
- Deploy model selector script on head node
- Update head node service to support dynamic model routing

### Phase 3: Multi-Model Service Deployment
- Deploy RPC worker service with Strand 14B on Node 2
- Deploy RPC worker service with DeepSeek V3 on Node 3
- Deploy updated multi-model head service on Node 1

### Phase 4: Launch & Verification
- Launch RPC workers (different models)
- Launch head node with model routing
- Test all three models independently
- Verify distributed GPU usage

### Phase 5: Multi-Model Performance Testing
- Baseline OR1-Behemoth 73B (deep analysis tasks)
- Baseline Strand-Rust-Coder 14B (idiomatic analysis)
- Baseline DeepSeek Coder V3 (complex ownership tasks)

## Comparison: Original vs Revised Strategy

| Aspect | Original Plan | Revised Strategy |
|---------|---------------|-----------------|
| **Primary Model** | OR1-Behemoth 73B Q8 (77GB) | OR1-Behemoth 73B Q8 (77GB) |
| **Node 2 Model** | Same as primary (OR1) | Strand-Rust-Coder 14B Q8 (7GB) |
| **Node 3 Model** | Same as primary (OR1) | DeepSeek Coder V3 671B Q5 (~120GB) |
| **Total VRAM** | 77GB + 77GB + 77GB = 231GB (doesn't fit) | 77GB + 7GB + 120GB = 204GB (fits) |
| **Task Types** | One model for all tasks | Task-specific model routing |
| **Idiomatic Analysis** | OR1 (larger context, but coherence drift) | Strand (94.3% compile rate, peer-reviewed) |
| **Compilation Safety** | OR1 (~70-90% success) | Strand (94.3% success, higher reliability) |
| **Complex Refactoring** | OR1 (larger context) | DeepSeek V3 (specialized in ownership puzzles) |
| **Hardware Flexibility** | Requires 3×77GB (impossible) | Strand runs on consumer GPUs |
| **Deployment Time** | Single model (25 min) | Three models (55-70 min) |

## Recommendation

**Proceed with hybrid strategy** for these reasons:

1. **Task Optimization**: Each model excels at specific use cases
2. **Hardware Fit**: All models fit comfortably in available VRAM
3. **Risk Reduction**: Diverse models reduce single-point failure
4. **Flexibility**: Easy to adapt to future models or new use cases
5. **Benchmark Data**: Strand has proven Rust performance (94.3% compile rate)

## Next Steps

1. Update production guide with hybrid deployment steps
2. Update beads epic with revised tasks
3. Begin Phase 1: Multi-model download
4. Deploy model router script
5. Launch and verify all three models

---

**Created**: January 16, 2026
**Based on**: "State of Art in Agentic Rust Development: A Deep Analysis of Specialized Architectures" (independent analysis, 2026)
**Status**: Plan Updated - Ready for Implementation
