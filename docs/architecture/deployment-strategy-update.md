# Deployment Plan Updated - Hybrid Model Strategy

**Date**: January 16, 2026
**Status**: ✅ Strategy Updated

## Summary

Updated deployment strategy based on comprehensive analysis of 2026 agentic Rust coding models. Shifted from single-model (OR1-Behemoth 73B) to **hybrid multi-model approach** optimized for different task types.

## What Changed

### Original Plan
- **Single Model**: OR1-Behemoth 73B Q8_0 (77 GB)
- **All nodes**: Same model on all 3 nodes
- **Hardware**: Requires 77GB VRAM on head node (tight fit)

### Revised Plan (Hybrid Strategy)
- **Primary Model**: OR1-Behemoth 73B Q8_0 (77 GB) - Node 1
- **Secondary Models**:
  - Strand-Rust-Coder 14B Q8_0 (7 GB) - Node 2
  - DeepSeek Coder V3 671B Q5 (120 GB) - Node 3
- **Model Router**: Script-based routing on head node
- **Hardware**: Fits comfortably (77GB + 7GB + 120GB = 204GB total)

## Why This Change?

### Key Findings from Analysis

1. **OR1-Behemoth Issues**:
   - "Embiggened" architecture (73B scaled from 32B)
   - Coherence drift and repetition loops
   - Stability concerns without extensive pre-training
   - Better suited for library code, not "deep analysis"

2. **Strand-Rust-Coder 14B Advantages**:
   - 94.3% compilation rate (peer-reviewed swarm training)
   - +13% improvement over larger models
   - Idiomatic Rust specialization (Typestate, generics, no_std)
   - Fast latency (~10 tokens/sec iteration)
   - Fits in 7GB VRAM (runs on consumer GPUs)

3. **DeepSeek Coder V3 Capabilities**:
   - 671B MoE with 37B active parameters
   - Self-correction mechanism (critical for borrow checker issues)
   - Massive knowledge base (System 2 reasoning)
   - Unmatched at solving novel ownership patterns

4. **Hardware Efficiency**:
   - Original: All 3 nodes need  accommodate 77GB (tight fit)
   - Revised: Head (77GB) + Node2 (7GB) + Node3 (120GB)
   - Node2 and Node3 can run on consumer GPUs if needed
   - Lower total cluster cost

## Deployment Architecture

### Node Roles

| Node | Primary Model | Secondary Models | Use Case |
|-------|---------------|------------------|-----------|
| **Node 1** (Head) | OR1-Behemoth 73B Q8 | N/A | Deep analysis, architectural refactoring, code completion |
| **Node 2** (RPC) | Strand-Rust-Coder 14B Q8 | Idiomatic analysis, real-time suggestions, typestate refactoring |
| **Node 3** (RPC) | DeepSeek Coder V3 671B Q5 | Complex ownership issues, novel patterns, borrow checker puzzles |

### Model Router (Node 1)

Script at `/usr/local/bin/llama-model-selector.sh` determines model based on task type:

```bash
TASK_TYPE="analyze"      → OR1-Behemoth (deep analysis)
TASK_TYPE="idiomatic"    → Strand-Rust-Coder (idiomatic improvements)
TASK_TYPE="refactor"     → Strand-Rust-Coder (idiomatic improvements)
TASK_TYPE="borrowing"    → DeepSeek Coder (ownership puzzles)
TASK_TYPE="ownership"    → DeepSeek Coder (ownership patterns)
TASK_TYPE="complex"      → DeepSeek Coder (complex ownership)
TASK_TYPE="fix"          → Strand-Rust-Coder (idiomatic fixes)
TASK_TYPE="complete"     → Strand-Rust-Coder (idiomatic fixes)
TASK_TYPE="typestate"    → Strand-Rust-Coder (idiomatic fixes)
```

### Service Architecture

**Head Node (Node 1)**:
- `/etc/systemd/system/llama-head.service` (updated)
- Dynamic model selection via router script
- Routes requests to appropriate RPC workers

**RPC Workers (Nodes 2 & 3)**:
- `/etc/systemd/system/llama-rpc.service` (different models)
- Node 2: Strand-Rust-Coder 14B Q8
- Node 3: DeepSeek Coder V3 671B Q5

## New Deployment Phases

### Phase 1: Multi-Model Acquisition (3 tasks)
1. **Download OR1-Behemoth 73B Q8** (primary model)
   - Stop Q4 download if running
   - Download Q8_0 (77 GB) to Node 1
   - Expected time: 20-25 minutes

2. **Download Strand-Rust-Coder 14B Q8** (Node 2)
   - Download 7 GB model
   - Expected time: 5 minutes

3. **Download DeepSeek Coder V3 Q5** (Node 3)
   - Download 120 GB model (may need multiple parts)
   - Expected time: 40-50 minutes

**Total Download Time**: ~65-80 minutes (for all 3 models)

### Phase 2: Model Router Configuration (1 task)
4. **Deploy model selector script** on Node 1
   - Create `/usr/local/bin/llama-model-selector.sh`
   - Configure environment variables for model routing
   - Test script with all task types

### Phase 3: Multi-Model Service Deployment (3 tasks)
5. **Deploy RPC worker with Strand 14B** on Node 2
   - Create systemd service for Strand model
   - Model path: `/opt/models/Strand-Rust-Coder-14B-v1.Q8_0.gguf`
   - Model size: 7 GB

6. **Deploy RPC worker with DeepSeek V3** on Node 3
   - Create systemd service for DeepSeek model
   - Model path: `/opt/models/DeepSeek-Coder-V3-Q5_K_M.gguf`
   - Model size: 120 GB
   - Note: Q5 quantization for 671B (vs Q8 for others)

7. **Deploy updated multi-model head service** on Node 1
   - Update `/etc/systemd/system/llama-head.service`
   - Use model router script in ExecStart
   - Support dynamic model selection

### Phase 4: Launch & Verification (4 tasks)
8. **Launch all three RPC workers**
   - Node 2: Strand model (7 GB)
   - Node 3: DeepSeek model (120 GB)
   - Verify services are running
   - Check GPU memory usage

9. **Launch head node with model routing**
   - Start multi-model head service
   - Verify model router works
   - Check logs for model selection

10. **Test API with OR1-Behemoth** (deep analysis)
   - Task type: "analyze"
   - Verify deep analysis capabilities
   - Measure throughput

11. **Test API with Strand-Rust-Coder** (idiomatic)
   - Task type: "idiomatic"
   - Verify idiomatic pattern detection
   - Measure real-time suggestions

12. **Test API with DeepSeek Coder** (complex)
   - Task type: "borrowing"
   - Verify ownership puzzle solving
   - Measure self-correction

### Phase 5: Multi-Model Performance Testing (6 tasks)
13. **Baseline OR1-Behemoth** (deep analysis tasks)
14. **Baseline Strand-Rust-Coder** (idiomatic tasks)
15. **Baseline DeepSeek Coder** (complex tasks)
16. **Compare throughput across all three models**
17. **Identify optimal model per task type**
18. **Document performance characteristics**

## Benefits of Hybrid Strategy

### 1. Task Optimization
- Right model for right job
- OR1: Deep analysis, large context
- Strand: Idiomatic analysis, fast iteration
- DeepSeek: Complex reasoning, novel patterns

### 2. Hardware Flexibility
- Strand (14B) fits in 7GB VRAM → consumer GPUs
- DeepSeek (671B) can be excluded if not needed
- Easy to scale: Add/remove models without infrastructure changes

### 3. Risk Mitigation
- Multiple models reduce single-point failure
- If OR1 hallucinates, Strand provides alternative
- Peer-reviewed Strand has higher reliability than embiggened OR1

### 4. Performance Data
- Each model optimized for specific use case
- Comparative analysis shows strengths/weaknesses
- Data-driven decisions for future model selection

## Comparison: Hardware Requirements

| Component | Original (OR1 ×3) | Revised (Hybrid) |
|-----------|---------------------|------------------|
| **Head Node VRAM** | 77 GB | 77 GB (same) |
| **Node 2 VRAM** | 77 GB | 7 GB (Strand) |
| **Node 3 VRAM** | 77 GB | 120 GB (DeepSeek) |
| **Total VRAM** | 231 GB (doesn't fit) | 204 GB (fits comfortably) |
| **Deployment Time** | 25-30 min | 65-80 min |

**Key Insight**: Revised plan uses less total VRAM while providing better task optimization.

## Model Selection Guide

When to use each model:

### OR1-Behemoth 73B
- **Use for**: Deep architectural analysis, code refactoring, complex logic
- **Strengths**: Large context, detailed explanations
- **Weaknesses**: Coherence drift, repetition, stability issues
- **Example**: "Analyze this crate's trait bounds and suggest generics"

### Strand-Rust-Coder 14B
- **Use for**: Idiomatic improvements, Typestate patterns, real-time completion
- **Strengths**: 94.3% compile rate, peer-reviewed, fast iteration
- **Weaknesses**: Limited context, smaller knowledge base
- **Example**: "Refactor this function to use Builder pattern with Typestate"

### DeepSeek Coder V3 671B
- **Use for**: Complex ownership issues, novel borrow checker puzzles, async problems
- **Strengths**: Self-correction, massive knowledge, unmatched reasoning
- **Weaknesses**: Slow (MoE routing), resource-intensive
- **Example**: "Solve this complex lifetime error that confuses standard models"

## Next Steps

1. ✅ **Update production guide** with hybrid deployment steps
2. ✅ **Create strategy document** (this file)
3. ✅ **Update beads epic** with revised tasks
4. 🔄 **Begin Phase 1** - Download all three models
5. 🔄 **Deploy model router** - Script for dynamic model selection
6. 🔄 **Deploy multi-model services** - All three models running
7. 🔄 **Test all three models** - Verify each excels at its use case
8. 🔄 **Document results** - Performance comparison and optimization

## Documentation

- **Production Guide**: `distributed-llama-production-guide.md` (updated)
- **Strategy Document**: `hybrid-model-strategy.md` (created)
- **Beads Epic**: `bd show beefcake2-lhr0` (updated)
- **Analysis Reference**: "State of Art in Agentic Rust Development: A Deep Analysis" (user-provided)

---

**Created**: January 16, 2026
**Status**: Strategy Updated, Ready for Implementation
**Total Estimated Time**: ~8-12 hours (including model downloads)
