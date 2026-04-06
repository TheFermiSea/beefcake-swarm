# Architecture

Key design decisions and rationale for the beefcake-swarm system.

## Core Design

Autonomous coding swarm: local LLM agents on an HPC cluster (3x V100S GPUs) coordinated through deterministic quality gates.

**Core flow:** Rig agents -> git worktree isolation -> Beads tracking -> SLURM dispatch

## Two-Crate Structure

- **coordination/** -- Deterministic logic layer (~21k LOC). NO LLM calls, pure state machines. Runs as MCP server (rmcp) exposing tools via stdio.
- **crates/swarm-agents/** -- Rig-based orchestrator. Cloud manager + local workers. The active loop that picks issues, dispatches work, and verifies results.

## Escalation Ladder

```
Cloud Manager (Claude Opus 4.6 via CLIAPIProxy, max 10 iterations)
    -> cloud fallback cascade (4 models)
    -> delegates to local workers on 3 GPU nodes
    -> verifier after each worker completes
    -> circuit breaker: 3 stuck iterations -> abort
    v
Human Intervention (blocking beads issue)
```

## Key Design Decisions

### Worktree Isolation
Each issue gets its own git worktree at `/tmp/beefcake-wt/<issue-id>`. This enables parallel issue processing without branch conflicts.

### Deterministic Verification
Quality gates (fmt, clippy, check, test) are deterministic -- no LLM involved. Pass/fail is binary and reproducible.

### Cloud-First Management
Cloud models manage from iteration 1. Local models are workers only. This ensures high-quality planning and delegation even when local models are weaker at reasoning.

### Stack Profiles
`SwarmStackProfile` controls role-to-model routing. Three profiles available: `hybrid_balanced_v1` (default), `small_specialist_v1`, `strategist_hybrid_v1`.

### Circuit Breaker
`SWARM_MAX_NO_CHANGE` (default: 3) aborts after consecutive iterations with no code changes. Prevents infinite loops when a model is stuck.
