# AI Inference Endpoints (2026 Dense Model Deployment)

Local LLM inference on the **Beefcake Swarm** cluster via llama.cpp. Migrated from MoE expert-offload to dense GPU-resident models for 7-21x throughput improvement.

## Quick Start

```bash
# Scout/Fast Tier (GLM-4.7-Flash, vasp-03)
# 30B/3B MoE with expert offload. Fast tool-calling and single-file fixes.
curl -sf http://vasp-03:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "Fix this borrow checker error..."}],
    "max_tokens": 1000
  }'

# Coder Tier (Qwen3.5-27B dense, vasp-01)
# Multi-file scaffolding and integration.
curl -sf http://vasp-01:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Refactor this module..."}],
    "max_tokens": 2000
  }'

# Reasoning Tier (Devstral-Small-2-24B dense, vasp-02)
# Complex reasoning and architecture decisions.
curl -sf http://vasp-02:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Devstral-Small-2-24B",
    "messages": [{"role": "user", "content": "Design the state machine for..."}],
    "max_tokens": 2000
  }'

# SWE Specialist (SERA-14B, vasp-03 port 8083)
# Software engineering agent from Allen AI (Qwen3 backbone).
curl -sf http://vasp-03:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SERA-14B",
    "messages": [{"role": "user", "content": "Implement the fix for..."}],
    "max_tokens": 2000
  }'
```

## Endpoints

| Tier | Endpoint | Hardware | Model | Quant | Throughput |
|------|----------|----------|-------|-------|------------|
| **Scout / Fast** | `http://vasp-03:8081` | V100S 32GB (expert-offload) | GLM-4.7-Flash (30B/3B MoE) | Q4_K_M | ~50 tok/s |
| **Coder** | `http://vasp-01:8081` | V100S 32GB (GPU-resident) | Qwen3.5-27B (dense) | Q4_K_M | ~27 tok/s |
| **Reasoning** | `http://vasp-02:8081` | V100S 32GB (GPU-resident) | Devstral-Small-2-24B (dense) | Q4_K_M | ~30 tok/s |
| **SWE Specialist** | `http://vasp-03:8083` | V100S 32GB (shared w/ Scout) | SERA-14B (Qwen3 backbone, Allen AI) | Q4_K_M | TBD |
| **Embedding** | `http://vasp-*:8082` | CPU only | nomic-embed-text-v1.5 | Q8_0 | <1ms |
| **Cloud** | `http://localhost:8317` | ai-proxy | gpt-5.4-mini (CLIAPIProxy) | N/A | N/A |

## Model Tiers

### Scout / Fast Tier (Tool-Calling Speed)

- **Model**: `GLM-4.7-Flash` (30B/3B MoE, Q4_K_M)
- **Architecture**: Mixture-of-Experts with 3B active parameters. Expert FFN layers offloaded to CPU, attention on GPU.
- **Node**: vasp-03 (shared with SWE Specialist on port 8083).
- **Throughput**: ~50 tok/s.
- **Use Case**: Tool-calling, single-file fixes, scout and reviewer roles.

### Coder Tier (Multi-File Integration)

- **Model**: `Qwen3.5-27B` (dense, Q4_K_M)
- **Architecture**: Dense 27B — fully GPU-resident on V100S 32GB.
- **Node**: vasp-01.
- **Throughput**: ~27 tok/s.
- **Use Case**: Multi-file code generation, complex refactoring, general worker.

### Reasoning Tier (Deep Analysis)

- **Model**: `Devstral-Small-2-24B` (dense, Q4_K_M)
- **Architecture**: Dense 24B from Mistral AI — fully GPU-resident on V100S 32GB.
- **Node**: vasp-02.
- **Throughput**: ~30 tok/s.
- **Use Case**: Complex reasoning, architecture decisions, planning, escalation fallback.

### SWE Specialist Tier

- **Model**: `SERA-14B` (Qwen3 backbone, Q4_K_M)
- **Architecture**: Dense 14B fine-tuned by Allen AI for software engineering tasks.
- **Node**: vasp-03 on port 8083 (shares GPU with Scout tier on 8081).
- **Throughput**: TBD.
- **Use Case**: Targeted SWE agent tasks via TensorZero routing experiments.

### Embedding Tier

- **Model**: `nomic-embed-text-v1.5` (Q8_0)
- **Architecture**: CPU-only embedding model. Runs on any vasp node port 8082.
- **Throughput**: <1ms per embedding.
- **Use Case**: Semantic search, CocoIndex, RAG retrieval.

### Cloud Tier

- **Manager**: `gpt-5.4-mini` via CLIAPIProxy (localhost:8317)
- **Fallback chain**: gpt-5.4-mini -> gemini-3.1-pro-high -> claude-sonnet-4-6 -> gemini-3.1-flash-lite-preview
- **Council**: Librarian=gemini-3.1-pro-preview, Architect=claude-opus-4-6, Strategist=gpt-5.2-codex

## Cluster Hardware

- 3x V100S 32GB nodes (vasp-01/02/03), ConnectX-6 100Gbps InfiniBand
- Each node: 256GB RAM, 32 cores
- Dense models are fully GPU-resident (no expert offload), except GLM-4.7-Flash (MoE with expert offload)
- TensorZero adaptive routing experiments run across all 4 local models via Thompson Sampling

## Health Checks

```bash
# All endpoints
curl -sf http://vasp-03:8081/health   # Scout (GLM-4.7-Flash)
curl -sf http://vasp-01:8081/health   # Coder (Qwen3.5-27B)
curl -sf http://vasp-02:8081/health   # Reasoning (Devstral-Small-2-24B)
curl -sf http://vasp-03:8083/health   # SWE Specialist (SERA-14B)
curl -sf http://vasp-03:8082/health   # Embedding (nomic-embed-text-v1.5)

# Quick all-node check
for h in vasp-01 vasp-02 vasp-03; do
  echo -n "$h:8081 "; curl -sf --max-time 2 http://$h:8081/health && echo "OK" || echo "DOWN"
done
```

## SLURM Management

AI tiers run as preemptible jobs under the `ai_opportunistic` QoS.

```bash
# Check running jobs
squeue --name=llama-*

# Start inference (independent instances)
ssh root@10.0.0.22 "bash /tmp/start-inference.sh"   # vasp-03 (GLM-4.7-Flash + SERA-14B)
ssh root@10.0.0.20 "bash /tmp/start-inference.sh"   # vasp-01 (Qwen3.5-27B)
ssh root@10.0.0.21 "bash /tmp/start-inference.sh"   # vasp-02 (Devstral-Small-2-24B)
```

## Migration Notes

Previous deployment used Qwen3.5-122B-A10B MoE (~1.5 tok/s actual, ~5-8 tok/s theoretical) on vasp-01/02 with expert offload. Migrated to dense models (Qwen3.5-27B, Devstral-Small-2-24B) for 7-21x throughput improvement. vasp-03 previously ran Qwen3-Coder-Next (80B/3B MoE), replaced by GLM-4.7-Flash (30B/3B MoE) for higher throughput.

---
**Updated**: March 29, 2026
**Status**: ACTIVE -- Dense model deployment with TensorZero routing
