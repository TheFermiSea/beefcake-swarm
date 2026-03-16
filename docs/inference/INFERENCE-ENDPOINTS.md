# AI Inference Endpoints (2026 Independent Instances)

Local LLM inference on the **Beefcake Swarm** cluster via llama.cpp.

## Quick Start (March 2026 Independent Instances)

```bash
# Scout/Fast Tier (27B Opus-Distilled, vasp-03)
# Distilled from Claude 4.6 Opus reasoning. Specialized for tool-calling and single-file fixes.
curl -sf http://vasp-03:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B-Opus-Distilled",
    "messages": [{"role": "user", "content": "Fix this borrow checker error..."}],
    "max_tokens": 1000
  }'

# Coder Tier (122B MoE, vasp-01)
# Optimized for multi-file scaffolding and integration.
curl -sf http://vasp-01:8081/v1/chat/completions ...

# Reasoning Tier (122B MoE, vasp-02)
# Complex reasoning and architecture decisions.
curl -sf http://vasp-02:8081/v1/chat/completions ...
```

## Endpoints

| Tier | Endpoint | Hardware | Model | Throughput |
|------|----------|----------|-------|------------|
| **Scout / Fast** | `http://vasp-03:8081` | V100S 32GB (VRAM-resident) | Qwen3.5-27B-Opus-Distilled Q4_K_M | ~34 tok/s |
| **Coder** | `http://vasp-01:8081` | V100S 32GB (expert-offload) | Qwen3.5-122B-A10B MoE | ~5-8 tok/s |
| **Reasoning** | `http://vasp-02:8081` | V100S 32GB (expert-offload) | Qwen3.5-122B-A10B MoE | ~5-8 tok/s |

## Model Tiers

### Scout / Fast Tier (Specialized Reliability)

- **Model**: `Qwen3.5-27B-Opus-Distilled` (Q4_K_M quantization)
- **Rationale**: Distilled from Claude 4.6 Opus reasoning, this model solves the "Turn-Based Trap" where workers provide analysis but fail to call tools.
- **Capacity**: VRAM-resident on a single V100S (32GB).
- **Context**: 65K tokens.
- **Throughput**: ~34 tok/s.

### Coder Tier (Multi-File Integration)

- **Model**: `Qwen3.5-122B-A10B-MoE`
- **Architecture**: Mixture-of-Experts with ~10B active parameters.
- **Optimization**: Expert-offload on vasp-01, single-node independent instance.
- **Context**: 65K tokens.
- **Throughput**: ~5-8 tok/s.
- **Use Case**: Multi-file code generation and complex refactoring.

### Reasoning Tier (Deep Analysis)

- **Model**: `Qwen3.5-122B-A10B-MoE`
- **Architecture**: Mixture-of-Experts with ~10B active parameters.
- **Optimization**: Expert-offload on vasp-02, single-node independent instance.
- **Context**: 65K tokens.
- **Throughput**: ~5-8 tok/s.
- **Use Case**: Complex reasoning, architecture decisions, escalation fallback.

### Cloud Tier

- **Manager**: `claude-opus-4-6` via CLIAPIProxy (localhost:8317)
- **Fallback chain**: opus-4-6 → gemini-3.1-pro-high → claude-sonnet-4-6 → gemini-3.1-flash-lite-preview
- **Validators**: gemini-3.1-pro-preview + claude-sonnet-4-6 + gpt-5.2-codex (3 concurrent)
- **Council**: Librarian=gemini-3.1-pro-preview, Architect=claude-opus-4-6, Strategist=gpt-5.2-codex

## SLURM Management

AI tiers run as preemptible jobs under the `ai_opportunistic` QoS.

```bash
# Check running jobs
squeue --name=llama-27b,llama-122b

# Start inference (independent instances)
ssh root@10.0.0.22 "sbatch /cluster/shared/scripts/run-27b-256k.slurm"
ssh root@10.0.0.20 "sbatch /cluster/shared/scripts/run-122b-rpc.slurm"
```

---
**Updated**: March 16, 2026
**Status**: ACTIVE — Independent instances deployed
