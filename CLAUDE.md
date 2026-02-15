# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Selection & Routing
- **Rust Tasks:** Use the `/ask-local` command (wraps `or1-behemoth-q4_k_m.gguf`) for deep Rust analysis and generation.
  - Example: `/ask-local "or1-behemoth-q4_k_m.gguf" "Explain the borrow checker error in src/lib.rs"`
- **Code Gen:** Use the `/ask-local` command (wraps `Qwen3-Coder-Next`) for scaffolding and boilerplate.
  - Example: `/ask-local "Qwen3-Coder-Next" "Generate a struct for User with fields..."`
- **General/Complex:** Use the default Claude model (Sonnet 3.5 / Opus).
- **Research:** Always check **NotebookLM** first using `/ask-notebook` or `notebook_query`.

## Build & Test Commands

```bash
cargo build --workspace                    # Build all crates
cargo test -p coordination                 # Run coordination tests
cargo test -p coordination <test_name>     # Run single test
cargo test -p swarm-agents                 # Run swarm-agents tests
cargo run -p swarm-agents                  # Run orchestrator (needs inference running)
cargo run -p coordination                  # Run MCP server (stdio transport)
cargo run -p coordination -- --harness     # MCP server with harness tools
cargo run -p coordination -- --ensemble --state-path ./state  # MCP with ensemble
cargo fmt --all -- --check                 # Check formatting
cargo clippy --workspace -- -D warnings    # Lint
```

## Architecture

Autonomous coding swarm: local LLM agents on an HPC cluster (3x V100S GPUs) coordinated through deterministic quality gates.

**Core flow:** `Rig agents → Gastown worktrees → Beads tracking → SLURM dispatch`

### Workspace Crates

**`coordination/`** — Deterministic logic layer (~21k LOC). NO LLM calls, pure state machines. Runs as an MCP server (rmcp) exposing tools via stdio.

Key modules:
- `verifier/` — Quality gate pipeline: `cargo fmt` → `clippy` → `cargo check` (JSON) → `cargo test`. Parses rustc error categories (borrow checker, lifetimes, trait bounds, type mismatch, async/Send).
- `escalation/` — Tier routing state machine. Triggers: error repeat 2x → escalate, >3 compile failures → escalate, >8 files changed → Cloud.
- `feedback/` — Compilation error correction loop with tiered escalation. Runs compiler, parses errors, routes to appropriate model tier.
- `ensemble/` — Multi-model coordination: submit task → execute all models sequentially (load/run/store/unload) → voting (majority/weighted/unanimous) → arbitration if tie. Uses RocksDB for state persistence.
- `council/` — Cloud AI escalation adapter. Three members: Librarian (Gemini 3 Pro), Architect (Claude Sonnet 4), Manager (GPT-4o). Queries concurrently, synthesizes weighted decision.
- `harness/` — Agent session management (Anthropic patterns): session state persistence, git checkpoints/rollback, feature registry, sub-session delegation, human intervention requests.
- `slurm/` — SLURM inference lifecycle (1.3k LOC): job submission, health checks (TCP+HTTP), endpoint discovery via NFS JSON, recovery state machine, preemption handling.
- `router/` — Task classification: type mismatch/imports → Strand (fast), borrow/lifetimes/traits → Hydra → OR1, complex/multi-file → OR1 (reasoning).
- `work_packet/` — Compact context handoffs between tiers (vs full transcript). Includes task, file contexts with key symbols, error history, constraints.
- `events/` — Pub/sub event bus for ensemble session tracking and replay.
- `state/` — RocksDB-backed persistent store for ensemble sessions, tasks, votes.

**`crates/swarm-agents/`** — Rig-based orchestrator for the 2-agent loop. Currently Phase 1 (beads connectivity). The planned MVP loop:
1. Pick highest-priority beads issue
2. Create Gastown worktree
3. Implementer (72B) writes code
4. Verifier (deterministic gates)
5. Validator (14B, blind review — no implementer context)
6. Pass → merge + close issue; Fail → update notes, loop

### Escalation Ladder

```
Cloud Manager (Opus 4.6 via CLIAPIProxy, 10 iterations)
    → delegates to local workers: OR1-Behemoth (reasoning), strand-14B (Rust), Qwen3-Coder-Next (general)
    → runs verifier after each worker completes
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

Cloud models are the managers from iteration 1. Local models are workers.
When cloud is unavailable, OR1-Behemoth serves as fallback local manager.

### Non-Workspace Directories

- `flywheel/` — Forked from `Dicklesworthstone/agentic_coding_flywheel_setup`. TypeScript/Node. Mining prompts and task decomposition strategies; discarding Docker/cloud, adapting to SLURM/NFS.
- `indexing/` — Python scripts for code indexing (CocoIndex for semantic search/RAG).
- `inference/` — SLURM job scripts (`run-14b.slurm` serves strand-14B + Qwen3-Coder-Next on vasp-02, `run-72b-distributed.slurm` on vasp-01+03), systemd daemon, build/validate scripts.
- `infrastructure/` — Monitoring: GPU dashboard, HPC watchdog, ai-proxy setup.
- `docs/` — Architecture docs, deployment guides, inference endpoint specs.

## Inference Endpoints (must be running via SLURM)

| Tier | Endpoint | Model | Throughput |
|------|----------|-------|------------|
| Fast (14B) | http://vasp-02:8080 | strand-rust-coder-14b-q8_0 | ~53 tok/s |
| Coder (80B MoE) | http://vasp-02:8080 | Qwen3-Coder-Next | ~5-15 tok/s |
| Reasoning (72B) | http://vasp-01:8081 | or1-behemoth-q4_k_m | ~13 tok/s |

Role specialization:
- **strand-14B** = "Mechanic" — fast Rust-specific fixes, borrow checker cascades, type errors
- **Qwen3-Coder-Next** = "Implementer" — general coding, multi-file changes, 256K context (MoE offload to CPU)
- **OR1-Behemoth** = "Architect" — complex reasoning, architecture decisions

Start inference:
```bash
ssh root@10.0.0.5 "sbatch /cluster/shared/scripts/llama-cpp/run-14b.slurm"
ssh root@10.0.0.5 "sbatch /cluster/shared/scripts/llama-cpp/run-72b-distributed.slurm"
```

## External Tools (install separately)

- `br` (beads_rust): `cargo install --git https://github.com/Dicklesworthstone/beads_rust` — Binary-only CLI, NOT a Rust library. Must invoke via subprocess (see `beads_bridge.rs`).
- `bv` (beads_viewer): `go install github.com/Dicklesworthstone/beads_viewer@latest`
- `gastown`: `go install github.com/steveyegge/gastown@latest` — Git worktree isolation per agent task.

## Environment Variables

**Coordination MCP:**
- `ROUTER_URL` — LLM endpoint (default: `http://10.0.0.31:8000/v1/chat/completions`)
- `ARCHITECT_MODEL`, `CODER_MODEL`, `HYDRA_MODEL` — model name overrides
- `HARNESS_MAX_ITERATIONS`, `HARNESS_FEATURES_PATH`, `HARNESS_PROGRESS_PATH`

**SLURM:**
- `SLURM_SCRIPTS_PATH` (default: `/cluster/shared/scripts/llama-cpp`)
- `SLURM_ENDPOINTS_PATH` (default: `/cluster/shared/ai/endpoints`)
- `SLURM_HOST` (default: `slurm-ctl`)

**Council (cloud escalation):**
- `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

## Cluster Access

- slurm-ctl: `ssh root@10.0.0.5` (controller, NFS server)
- vasp-01: `ssh root@10.0.0.20` (72B head, V100S)
- vasp-02: `ssh root@10.0.0.21` (14B fast, V100S)
- vasp-03: `ssh root@10.0.0.22` (72B RPC worker, V100S)
- ai-proxy: `ssh root@100.105.113.58` (external gateway LXC)

## SLURM Rules

**ALL computational tasks MUST go through SLURM.** Never run workloads directly on compute nodes.

## NFS Layout

```
/cluster/shared/
├── llama-cpp/bin/        # Inference binaries
├── scripts/llama-cpp/    # SLURM job scripts
├── ai/endpoints/         # Service discovery JSON
├── ai/logs/              # Shared logs
└── (future)
    ├── gastown-town/     # Gastown workspace root
    └── beads-db/         # Shared beads database
```

## Known Issues

- `coordination/tests/` — Several integration tests reference `rust_cluster_mcp` as an unresolved crate (should be `coordination`). These tests won't compile until import paths are fixed.
- `crates/swarm-agents/` — Has dead code warnings on structs/methods that are defined but not yet wired into the Phase 2 orchestrator loop.
- `#![allow(dead_code)]` is enabled in coordination's `lib.rs` and `main.rs` due to rmcp macro-generated code triggering false positives.

## Agent Teams

Enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in `.claude/settings.json`.

### Team Structure
- **Lead (Opus 4.6)**: Picks beads issues, assigns to teammates, reviews results
- **Teammates (Sonnet 4.5)**: Each works on one beads issue on a separate branch

### Teammate Workflow
1. Claim issue: `bd update <id> --status in_progress`
2. Create branch: `git checkout -b swarm/<issue-id>`
3. Implement the fix/feature
4. Quality gates auto-run on task completion (fmt, clippy, check, test)
5. Commit with conventional format and push branch

### Local Model Access (optional)
Teammates can query local Rust-expert models via curl for a second opinion:
- strand-14B (fast fixes): `curl http://vasp-02:8080/v1/chat/completions -d '{"model":"strand-rust-coder-14b-q8_0.gguf",...}'`
- OR1-Behemoth (reasoning): `curl http://vasp-01:8081/v1/chat/completions -d '{"model":"or1-behemoth-q4_k_m.gguf",...}'`

### Branch Strategy
Each teammate works on `swarm/<issue-id>`. Lead assigns non-overlapping issues.
