> **HISTORICAL DOCUMENT** (as of March 16, 2026): This plan has been superseded. The 397B model
> was replaced by independent 122B-A10B instances on vasp-01 and vasp-02, and the 27B Scout model
> is now Qwen3.5-27B-Opus-Distilled (distilled from Claude 4.6 Opus reasoning). See `CLAUDE.md`
> for the current model roster and escalation ladder.

# 2026 Model Ensemble Upgrade - Qwen3.5 Hybrid Strategy

**Date**: March 9, 2026
**Status**: HISTORICAL (superseded by independent instance deployment)

## Overview

Based on production failure analysis from Jan-Feb 2026, the **Beefcake Swarm** is transitioning from a homogenous Qwen3.5-397B deployment to a **task-specialized hybrid ensemble**. This upgrade addresses the "Turn-Based Trap" (where large models provide analysis but fail to call tools) and optimizes for the V100S cluster's memory/throughput constraints.

## The New Ensemble Stack

| Tier | Model | Hardware | Active Params | Key Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Fast / Scout** | **Qwen3.5-27B-Distilled-Reasoning** | `vasp-03` | 27B | High reliability for `edit_file` calls. Fast TTFT. |
| **Integrator** | **Qwen3.5-122B-A10B-MoE** | `vasp-01` | ~10B | Optimal throughput/quality balance for multi-file tasks. |
| **The Council** | **Qwen3.5-397B-A17B-MoE** | `vasp-02` | ~17B | Maximum reasoning depth for architectural arbitration. |

## Rationale for Shift

### 1. Reliability (The "Mechanic" Problem)
Current logs with Qwen3.5-397B as the primary worker showed a 12% success rate in dogfood runs. The model often generated massive textual analysis but exited the Rig loop before calling `edit_file`. The **Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled** model is specifically trained to solve this by distilling the tool-calling precision of larger reasoning models into a smaller, faster footprint.

### 2. Throughput (V100S Optimization)
The 397B model requires massive CPU-offloading for its MoE experts on our V100S nodes, capping generation at ~8 tokens/sec. 
- The **122B-A10B** model allows a higher percentage of weights to stay in the GPU/Memory path, significantly increasing integration speed.
- The **27B** model fits entirely within the 32GB VRAM of a single V100S, enabling blazing fast single-file repairs.

## Node Assignment (3-Node Cluster)

| Node | Model Role | Primary Model | Hardware Load |
| :--- | :--- | :--- | :--- |
| **vasp-01** | Integrator | Qwen3.5-122B-A10B | ~80GB RAM (Hybrid CPU/GPU) |
| **vasp-02** | The Council | Qwen3.5-397B-A17B | ~240GB RAM (MoE Offloading) |
| **vasp-03** | Fast / Scout | Qwen3.5-27B-Distilled | ~32GB VRAM (Full GPU Offload) |

## Implementation Roadmap (Beads Epic: `beefcake-model-upgrade-2026`)

### Phase 1: Model Acquisition
- [ ] Download `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` to `vasp-03`.
- [ ] Download `Qwen3.5-122B-A10B-GGUF` to `vasp-01`.
- [ ] Validate GGUF checksums and quantizations (Q4_K_M preferred).

### Phase 2: SLURM Infrastructure
- [ ] Create `inference/slurm/run-27b-distilled.slurm` with full GPU offloading.
- [ ] Create `inference/slurm/run-122b-moe.slurm` optimized for vasp-01 memory topology.
- [ ] Update `ai-proxy` health checks for the new tier structure.

### Phase 3: Orchestrator Refactoring
- [ ] Refactor `crates/swarm-agents/src/config.rs` to use environment-driven aliases for tiers.
- [ ] Update `crates/swarm-agents/src/prompts.rs` to optimize preambles for distilled reasoning vs MoE integrator.

### Phase 4: Verification
- [ ] Benchmarking compilation rates on `RustEvo²` dataset.
- [ ] Verify `edit_file` reliability with the 27B distilled model.
- [ ] Load-test 3-node concurrent inference.

---
**Updated**: March 9, 2026
**Reference**: `agent-harness-survey.md`
