# AI Inference Endpoints (2026 Hybrid Ensemble)

Local LLM inference on the **Beefcake Swarm** cluster via llama.cpp.

## Quick Start (March 2026 Tiered Ensemble)

```bash
# Fast/Scout Tier (27B Distilled, vasp-03)
# Specialized for tool-calling and single-file fixes.
curl -sf http://vasp-03:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B-Distilled",
    "messages": [{"role": "user", "content": "Fix this borrow checker error..."}],
    "max_tokens": 1000
  }'

# Integrator Tier (122B MoE, vasp-01)
# Optimized for multi-file scaffolding and integration.
curl -sf http://vasp-01:8081/v1/chat/completions ...

# The Council (397B MoE, vasp-02)
# Complex architecture decisions and final arbitration.
curl -sf http://vasp-02:8081/v1/chat/completions ...
```

## Endpoints

| Tier | Endpoint | Hardware | Model Role |
|------|----------|----------|------------|
| **Fast / Scout** | `http://vasp-03:8081` | V100S 32GB (Full Offload) | Reliability, Tool-Use |
| **Integrator** | `http://vasp-01:8081` | 80GB RAM + GPU | Scaffold, Integration |
| **The Council** | `http://vasp-02:8081` | 240GB RAM (MoE Offload) | Reasoning, Arbitration |

## Model Tiers

### Fast / Scout Tier (Specialized Reliability)

- **Model**: `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`
- **Rationale**: Distilled from Claude 4.6 Reasoning, this model solves the "Turn-Based Trap" where workers provide analysis but fail to call tools.
- **Capacity**: Full GPU offload on a single V100S (32GB).
- **Throughput**: ~25-35 tok/s.

### Integrator Tier (Throughput Optimized)

- **Model**: `Qwen3.5-122B-A10B-MoE`
- **Architecture**: Mixture-of-Experts with ~10B active parameters.
- **Optimization**: Optimized for vasp-01 NUMA topology. Higher active parameter percentage in VRAM vs 397B.
- **Use Case**: Multi-file code generation and complex refactoring.

### The Council (Maximum Reasoning)

- **Model**: `Qwen3.5-397B-A17B-MoE`
- **Architecture**: Massive MoE (~17B active) with heavy CPU expert offloading.
- **Performance**: ~8-10 tok/s.
- **Use Case**: Escalation fallback for when Integrator or Fast tiers are stuck.

## SLURM Management

AI tiers run as preemptible jobs under the `ai_opportunistic` QoS.

```bash
# Check running jobs
squeue --name=llama-27b,llama-122b,llama-397b

# Start new ensemble
sbatch /cluster/shared/scripts/run-27b-distilled.slurm
sbatch /cluster/shared/scripts/run-122b-moe.slurm
sbatch /cluster/shared/scripts/run-qwen35.slurm
```

---
**Updated**: March 9, 2026
**Status**: ENSEMBLE TRANSITION IN PROGRESS
