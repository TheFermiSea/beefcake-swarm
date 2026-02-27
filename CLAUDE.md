# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Selection & Routing
- **Rust/Code Tasks:** Use `/ask-local` with the Qwen3.5-397B instances for Rust analysis, code generation, and architecture.
  - Architect (vasp-01): `/ask-local "Qwen3.5-397B-A17B" "Explain the borrow checker error in src/lib.rs"`
  - Implementer (vasp-02): `/ask-local "Qwen3.5-397B-A17B" "Generate a struct for User with fields..."`
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
    → delegates to local workers: Qwen3.5-397B Architect (vasp-01), Qwen3.5-397B Implementer (vasp-02)
    → runs verifier after each worker completes
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

Cloud models are the managers from iteration 1. Local models are workers.
When cloud is unavailable, Qwen3.5-397B Architect (vasp-01) serves as fallback local manager.

### Non-Workspace Directories

- `flywheel/` — Forked from `Dicklesworthstone/agentic_coding_flywheel_setup`. TypeScript/Node. Mining prompts and task decomposition strategies; discarding Docker/cloud, adapting to SLURM/NFS.
- `indexing/` — Python scripts for code indexing (CocoIndex for semantic search/RAG).
- `inference/` — SLURM job scripts (`run-qwen35.slurm` serves Qwen3.5-397B independently on vasp-01 and vasp-02), systemd daemon, build/validate scripts.
- `infrastructure/` — Monitoring: GPU dashboard, HPC watchdog, ai-proxy setup.
- `docs/` — Architecture docs, deployment guides, inference endpoint specs.

## Inference Endpoints

| Tier | Endpoint (Rig uses this) | Model | Throughput |
|------|--------------------------|-------|------------|
| All local tiers | http://vasp-02:8080/v1 | HydraCoder 30B-A3B MoE (Q4_K_M) | ~135 tok/s gen |

**Current setup**: HydraCoder on vasp-02:8080 (all tiers). 1 slot @ 32K context.

**Q4_K_M download (in progress)**: Qwen3.5-397B-A17B Q4_K_M from lmstudio-community (241GB, 7 shards).
Shards 6+7 complete on vasp-02 at `/scratch/ai/models/lmstudio-Qwen3.5-397B-A17B-GGUF/`.
Shards 1-5 downloading via wget (~197GB, ETA ~18-20hrs at 2.35MB/s).
Monitor: `ssh root@10.0.0.21 tail -f /tmp/q4km-download.log`
vasp-02 has 249GB free — sufficient. Once complete, build and serve with `--override-tensor exps=CPU`.

**UD-Q4_K_XL (broken)**: Exists on vasp-01 and vasp-03 (206GB). Confirmed broken for instruction
following — returns garbled output or immediate EOS regardless of prompt format. Do NOT use.
Tracked: beefcake-7v67.

**vasp-03 native llama.cpp build**: Rocky 8.8/GCC 8.5/CUDA 12.6 (GLIBC 2.28 compatible).
Binary: `/usr/local/bin/llama-server-vasp03`. Startup: `/tmp/start-qwen35.sh`.
Build script: `/tmp/build-qwen-llama.sh`. CUDA wrapper at `/usr/local/cuda/bin/nvcc`.

Role specialization (planned, once Q4_K_M ready):
- **vasp-02** = Primary — Qwen3.5-397B-A17B Q4_K_M, 4 slots @ 8K
- **vasp-03** = RPC GPU worker (32GB V100S) for vasp-02, or standalone with UD-Q4_K_XL
- **vasp-01** = Available (V100S + 256GB RAM), /scratch full (400GB)

Start inference:
```bash
# HydraCoder on vasp-02 (current, all tiers)
ssh root@10.0.0.21 "nohup /tmp/start-hydracoder.sh > /tmp/hydracoder-server.log 2>&1 &"
```

To switch swarm to Qwen3.5 endpoint once Q4_K_M is ready (no code change — env vars):
```bash
export SWARM_FAST_URL=http://vasp-02:8080/v1
export SWARM_CODER_URL=http://vasp-02:8080/v1
export SWARM_REASONING_URL=http://vasp-02:8080/v1
export SWARM_FAST_MODEL=Qwen3.5-397B-A17B
export SWARM_CODER_MODEL=Qwen3.5-397B-A17B
export SWARM_REASONING_MODEL=Qwen3.5-397B-A17B
export SWARM_LOCAL_BASE_URL=http://vasp-02:8080/v1
```

## External Tools (install separately)

- `bd` (beads): `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash` — Go binary, issue tracker CLI. Invoked via subprocess (see `beads_bridge.rs`).
- `bv` (beads_viewer): `go install github.com/Dicklesworthstone/beads_viewer@latest`
- `gastown`: `go install github.com/steveyegge/gastown@latest` — Git worktree isolation per agent task.
- `nlm` (notebooklm-mcp-cli): `uv tool install notebooklm-mcp-cli` — NotebookLM CLI for knowledge base queries. Auth: `nlm login`.

## NotebookLM Knowledge Base

The swarm uses NotebookLM as an external RAG layer for institutional memory. Complements CocoIndex (code structure) and Beads (issue tracking).

**Notebook Registry:** `notebook_registry.toml` maps roles to notebook IDs.

| Role | Notebook | Auto-Query | Purpose |
|------|----------|------------|---------|
| `project_brain` | beefcake-swarm: Project Brain | Yes | Architecture decisions, implementation summaries |
| `debugging_kb` | beefcake-swarm: Debugging KB | Yes | Error patterns, known fixes, resolution playbooks |
| `codebase` | beefcake-swarm: Codebase | No | Repomix-packed code for structural queries |
| `research` | beefcake-swarm: Research | No | Library docs, migration guides |
| `security` | beefcake-swarm: Security | No | OWASP, Rust security best practices |
| `visuals` | beefcake-swarm: Visuals | No | Dependency graphs, architecture diagrams |

**Query commands:**
```bash
nlm query notebook "<ID>" "What is the escalation ladder?"
nlm source add "<ID>" --file "doc.md"
```

**Orchestrator integration points:**
1. Pre-task: queries Project Brain for architectural context; on retries queries Debugging KB
2. Pre-escalation: checks Debugging KB for known fixes before escalating tier
3. Post-success: uploads resolution summary to Project Brain; tricky bugs (3+ iterations) also go to Debugging KB
4. Manager tool: `query_notebook` Rig tool available to cloud and local managers

**Complementary tool boundaries:**

| Tool | Scope |
|------|-------|
| CocoIndex | Code structure — callers, implementors, file navigation |
| NotebookLM | Knowledge — decisions, patterns, docs, error playbooks |
| Beads | Issue tracking — what needs to be done |
| Repomix | Feeds NotebookLM with packed codebase context |

**Environment:** `SWARM_NLM_BIN` overrides the `nlm` binary name (default: `"nlm"`).

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

- slurm-ctl: `ssh root@10.0.0.5` (controller, NFS server — VM 500 on pve1)
- vasp-01: `ssh root@10.0.0.20` (V100S + 256GB RAM, /scratch full — VM 600 on pve1)
- vasp-02: `ssh root@10.0.0.21` (V100S + 256GB RAM, HydraCoder running — VM 601 on pve2)
- vasp-03: `ssh root@10.0.0.22` (V100S + 256GB RAM — VM 602 on pve3)
- pve1: `ssh root@10.0.0.1` (Proxmox host, cluster gateway — DO NOT reboot)
- pve2: `ssh root@10.0.0.2` (Proxmox host)
- pve3: `ssh root@10.0.0.3` (Proxmox host)
- ai-proxy: `ssh brian@100.105.113.58` or `ssh root@100.105.113.58` (LXC on pve3)
  - Codebases live under `/home/brian/code/` (beefcake-swarm, rust-daq)
  - Use `brian` user for code work; `root` for system admin only
  - GitHub auth: SSH key (`ai-proxy-lxc`) + `gh` CLI as TheFermiSea

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
- `pve1 ZFS pool at 96%` — Deleted a 104GB stale `@restore` snapshot but pool is still 96% used. vasp-01's /scratch is 100% full (400GB). Consider cleaning old models (UD-Q4_K_XL, glm-4.7) from vasp-01.
- `vasp-03 NFS` — /home, /cluster/shared still NFS-mounted from slurm-ctl. Set `HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache` before running anything that writes to $HOME.

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
Teammates can query local Qwen3.5-397B instances via curl for a second opinion:
- Architect (vasp-01): `curl http://vasp-01:8081/v1/chat/completions -d '{"model":"Qwen3.5-397B-A17B",...}'`
- Implementer (vasp-02): `curl http://vasp-02:8080/v1/chat/completions -d '{"model":"Qwen3.5-397B-A17B",...}'`

### Branch Strategy
Each teammate works on `swarm/<issue-id>`. Lead assigns non-overlapping issues.
