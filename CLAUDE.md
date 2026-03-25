# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Selection & Routing

- **Scout/Fast (vasp-03:8081):** Qwen3-Coder-Next — 80B/3B MoE, expert-offload, 65K context. Used for scout, reviewer, fixer roles.
- **Coder (vasp-01:8081):** Qwen3.5-122B-A10B MoE — expert-offload, 65K context, ~5-8 tok/s. Multi-file code generation and integration.
- **Reasoning (vasp-02:8081):** Qwen3.5-122B-A10B MoE — expert-offload, 65K context, ~5-8 tok/s. Complex reasoning, planning, architecture decisions.
- **Cloud Manager:** Claude Opus 4.6 via CLIAPIProxy (localhost:8317). Fallback cascade: Gemini 3.1 Pro → Sonnet 4.6 → Gemini 3.1 Flash Lite.
- **Strategist (optional):** Qwen3.5-397B-A17B — advisor tier for non-writing arbitration. Configured via `SWARM_STRATEGIST_URL`.
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

**Core flow:** `Rig agents → git worktree isolation → Beads tracking → SLURM dispatch`

### Workspace Crates

**`coordination/`** — Deterministic logic layer (~21k LOC). NO LLM calls, pure state machines. Runs as an MCP server (rmcp) exposing tools via stdio.

Key modules:
- `verifier/` — Quality gate pipeline: `cargo fmt` → `clippy` → `cargo check` (JSON) → `cargo test`. Parses rustc error categories (borrow checker, lifetimes, trait bounds, type mismatch, async/Send).
- `escalation/` — Tier routing state machine. Triggers: error repeat 2x → escalate, >3 compile failures → escalate, >8 files changed → Cloud.
- `feedback/` — Compilation error correction loop with tiered escalation. Runs compiler, parses errors, routes to appropriate model tier.
- `ensemble/` — Multi-model coordination: submit task → execute all models sequentially (load/run/store/unload) → voting (majority/weighted/unanimous) → arbitration if tie. Uses RocksDB for state persistence.
- `council/` — Cloud AI escalation adapter. Three members: Librarian (Gemini 3.1 Pro), Architect (Claude Opus 4.6), Strategist (GPT-5.2 Codex). Queries concurrently, synthesizes weighted decision.
- `harness/` — Agent session management (Anthropic patterns): session state persistence, git checkpoints/rollback, feature registry, sub-session delegation, human intervention requests.
- `slurm/` — SLURM inference lifecycle (1.3k LOC): job submission, health checks (TCP+HTTP), endpoint discovery via NFS JSON, recovery state machine, preemption handling.
- `router/` — Task classification: categorizes errors by type (type mismatch/imports → fast tier, borrow/lifetimes/traits → coder tier, complex/multi-file → reasoning tier).
- `work_packet/` — Compact context handoffs between tiers (vs full transcript). Includes task, file contexts with key symbols, error history, constraints.
- `events/` — Pub/sub event bus for ensemble session tracking and replay.
- `state/` — RocksDB-backed persistent store for ensemble sessions, tasks, votes.
- `context_packer/` — AST-aware repo mapping, source providers, file walkers for building compact context.
- `benchmark/` — SLO/manifest/harness for benchmarking agents (includes GPU-accelerated physics gates).
- `analytics/` — Error tracking, replay, skill assessment, verification metrics.
- `debate/` — Consensus protocols, critique, guardrails, memory bridge, persistence.
- `memory/` — Budget tracking, compaction, summarization, observability.
- `reviewer_tools/` — AST-grep integration, graph RAG, rule packs for code review.
- `rollout/` — Feature flags for gradual capability rollout.
- `otel.rs` — OpenTelemetry tracing and metrics export.
- `shell_safety.rs` — Safe command execution with sandboxing.
- `patch.rs` — Git patch generation and application.
- `speculation.rs` — Proactive code analysis before errors occur.
- `resilience.rs` — Retry, backoff, circuit breaker patterns for external calls.

**`crates/swarm-agents/`** — Rig-based orchestrator (cloud manager + local workers). The active loop:
1. Pick highest-priority beads issue (or CLI `--issue`); up to `SWARM_PARALLEL_ISSUES` (default: 3) concurrently
2. Create git worktree in `/tmp/beefcake-wt/<issue-id>`
3. Cloud manager (Claude Opus 4.6 via CLIAPIProxy) plans and delegates
4. Local workers (27B on vasp-03, 122B on vasp-01/02) execute code changes
5. Verifier (deterministic quality gates) after each iteration — multi-language (Rust, Python, TypeScript, Go)
6. Pass → merge + close issue; Fail → retry up to `SWARM_MAX_RETRIES`; circuit breaker after `SWARM_MAX_NO_CHANGE` stuck iterations

Key modules:
- `agents/` — Cloud manager, coder, reviewer, specialist, adversary agents. Factory pattern with `SwarmRole`-based routing.
- `orchestrator/` — Main dispatch loop, validation, helpers.
- `modes/` — Runner architectures: contextual (Draft→Critique→Condense, NS-2), deepthink (JoinSet fan-out, NS-3), agentic (LLM-driven unified-diff, NS-4).
- `tools/` — 19 agent tools: colgrep, astgrep, exec, patch, fs, git, verifier, notebook, cargo_metadata, search_code, plan_parallel, workpad, apply_plan, bundles, proxy_wrappers, and more.
- `config.rs` — SwarmConfig, Tier, SwarmRole, SwarmStackProfile, CloudFallbackMatrix.
- `beads_bridge.rs` — Native `bd` subprocess integration.
- `file_targeting.rs` — Multi-language file targeting with snake_case/CamelCase extraction.
- `mutation_archive.rs` — Tracks successful mutations for feedback loops.
- `tensorzero.rs` — TensorZero feedback loop integration (experiment tracking, A/B testing).
- `telemetry.rs` — OpenTelemetry-compatible observability.

### Escalation Ladder

```text
Cloud Manager (Claude Opus 4.6 via CLIAPIProxy, max 10 iterations)
    → cloud fallback: Opus 4.6 → Gemini 3.1 Pro → Sonnet 4.6 → Gemini 3.1 Flash Lite
    → delegates to local workers:
        vasp-03:8081 — Qwen3-Coder-Next (scout, reviewer, fixer)
        vasp-01:8081 — Qwen3.5-122B-A10B MoE (coder, general worker)
        vasp-02:8081 — Qwen3.5-122B-A10B MoE (planner, reasoning worker)
    → optional: Qwen3.5-397B-A17B strategist (advisor, non-writing arbitration)
    → runs verifier after each worker completes
    → circuit breaker: SWARM_MAX_NO_CHANGE (default: 3) stuck iterations → abort
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

Cloud models are managers from iteration 1. Local models are workers.
When cloud is unavailable, falls back to worker-first mode (local models only).
Stack profile (`SWARM_STACK_PROFILE`) controls role→model routing: `hybrid_balanced_v1` (default), `small_specialist_v1`, `strategist_hybrid_v1`.

### Non-Workspace Directories

- `scripts/` — Orchestration scripts: `run-swarm.sh` (single-issue runner), `dogfood-loop.sh` (continuous loop), `benchmark-models.sh`, `deploy-dogfood.sh`, `postmortem-review.sh`, `generate-issues.sh`, `run-tz-evaluations.sh`, and more.
- `config/` — TensorZero configuration: `tensorzero.toml`, `functions/` (code_fixing, task_planning, cloud_manager_delegation, architect_plan, adversarial_test), `evaluations/` (manager delegation quality, code quality scoring).
- `inference/` — SLURM job scripts, systemd daemon, build/validate scripts for llama.cpp.
- `infrastructure/` — Monitoring: GPU dashboard, HPC watchdog, ai-proxy setup, cloud-proxy.service (socat relay), TensorZero docker-compose, scheduled benchmarking.
- `indexing/` — Python scripts for code indexing (CocoIndex for semantic search/RAG).
- `flywheel/` — Forked TypeScript/Node project for prompt mining and task decomposition strategies.
- `docs/` — Architecture docs, deployment guides, inference endpoint specs, dogfood diagnostics.

## Inference Endpoints

Heterogeneous local cluster — each node runs a different model tier. See `docs/inference/INFERENCE-ENDPOINTS.md` for the full reference.

| Tier | Endpoint | Model | Hardware | Throughput |
|------|----------|-------|----------|------------|
| Scout/Fast | http://vasp-03:8081/v1 | Qwen3-Coder-Next-UD Q4_K_XL (80B/3B MoE) | V100S 32GB (expert-offload) | TBD |
| Coder | http://vasp-01:8081/v1 | Qwen3.5-122B-A10B MoE | V100S 32GB (expert-offload) | ~5-8 tok/s |
| Reasoning | http://vasp-02:8081/v1 | Qwen3.5-122B-A10B MoE | V100S 32GB (expert-offload) | ~5-8 tok/s |
| Cloud | http://localhost:8317/v1 | claude-opus-4-6 (CLIAPIProxy) | ai-proxy | N/A |

**Qwen3-Coder-Next**: 80B/3B MoE (Mixture of Experts). Expert FFN layers offloaded to CPU RAM, attention on GPU. Replaces the 27B-Opus-Distilled as the fast tier model.

**122B-A10B MoE**: Mixture-of-Experts with ~10B active parameters. Expert FFN layers offloaded to CPU RAM (~225GB), attention on GPU. Each vasp node runs an independent instance.

**Cloud fallback matrix** (configured in config.rs `CloudFallbackMatrix::default_matrix`):
1. `claude-opus-4-6` (primary)
2. `gemini-3.1-pro-high` (fallback-1)
3. `claude-sonnet-4-6` (fallback-2)
4. `gemini-3.1-flash-lite-preview` (fallback-3)

**Start inference:**

```bash
ssh root@10.0.0.22 "bash /tmp/start-inference.sh"   # vasp-03 (Qwen3-Coder-Next)
ssh root@10.0.0.20 "bash /tmp/start-inference.sh"   # vasp-01 (122B-A10B coder)
ssh root@10.0.0.21 "bash /tmp/start-inference.sh"   # vasp-02 (122B-A10B reasoning)
```

Scripts are version-controlled at `inference/start-inference-*.sh` and deployed to `/tmp/start-inference.sh` on each node. vasp-01/02 use HPC SDK CUDA paths; vasp-03 uses cuda-unified.

**Cloud proxy:** CLIAPIProxy (Router-for-Me v6.8.54) runs on ai-proxy (localhost:8317). Binary at `/opt/cli-proxy-api/cli-proxy-api`, config at `/opt/cli-proxy-api/config.yaml`, credentials in `/root/.cli-proxy-api/`. Uses `x-api-key` header for inference, `Authorization: Bearer` for management API. API key: `rust-daq-proxy-key` (same for both, set via `SWARM_CLOUD_API_KEY` env var). Remote management enabled — accessible from any Tailnet host at `http://100.105.113.58:8317`. Use `/cloud-proxy` skill for diagnostics and management commands. Docs: https://help.router-for.me/management/api

**llama.cpp build:** v8231 (c024d8590), native on vasp-03 (Rocky 8.8/GCC 13/CUDA 12.6). Binary: `/usr/local/bin/llama-server-mmq` (compiled with `GGML_CUDA_FORCE_MMQ=ON` for V100), deployed to all 3 nodes. Includes autoparser refactor for Qwen3-Coder XML tool call parsing. NFS backup: `/cluster/shared/llama-cpp/bin/autoparser-build/`. Rollback: `/usr/local/bin/llama-server-mmq.b8179`.

## Swarm Environment Variables

Set by `scripts/run-swarm.sh` (overrides config.rs defaults). See `crates/swarm-agents/src/config.rs` for all options.

### Tier Endpoints

Heterogeneous model setup: 27B on vasp-03, 122B on vasp-01/02.

| Variable | Default |
|----------|---------|
| `SWARM_FAST_URL` | `http://vasp-03:8081/v1` |
| `SWARM_FAST_MODEL` | `Qwen3-Coder-Next` |
| `SWARM_CODER_URL` | `http://vasp-01:8081/v1` |
| `SWARM_CODER_MODEL` | `Qwen3.5-122B-A10B` |
| `SWARM_REASONING_URL` | `http://vasp-02:8081/v1` |
| `SWARM_REASONING_MODEL` | `Qwen3.5-122B-A10B` |
| `SWARM_STRATEGIST_URL` | *(none)* |
| `SWARM_STRATEGIST_MODEL` | `Qwen3.5-397B-A17B` |

### Cloud Endpoint

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_CLOUD_URL` | `http://localhost:8317/v1` (script) / *(none)* (config.rs) | Required for cloud manager mode |
| `SWARM_CLOUD_API_KEY` | *(none)* | Required if cloud URL set |
| `SWARM_CLOUD_MODEL` | `claude-opus-4-6` | Primary cloud model |
| `SWARM_CLOUD_FALLBACK_MODEL` (script) | `gemini-3.1-pro-high` | Shell-level fallback when primary model unavailable |
| `SWARM_CLOUD_FALLBACK_MODELS` (config.rs) | `claude-opus-4-6, gemini-3.1-pro-high, claude-sonnet-4-6, gemini-3.1-flash-lite-preview` | Comma-separated 4-model cascade in Rust |
| `SWARM_REQUIRE_ANTHROPIC_OWNERSHIP` | `1` | run-swarm.sh accepts both "anthropic" and "antigravity" |
| `SWARM_CLOUD_PREFLIGHT` | `1` | Probe cloud endpoint before starting |

### Behavior

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_MAX_RETRIES` | `6` | Max iterations per issue |
| `SWARM_CLOUD_MAX_RETRIES` | `3` | Cloud-specific retry limit |
| `SWARM_MAX_NO_CHANGE` | `3` | Circuit breaker: abort after N consecutive no-change iterations |
| `SWARM_CLOUD_ONLY` | `false` | Route all work through cloud (skip local) |
| `SWARM_VERIFIER_PACKAGES` | *(empty)* | Comma-separated; empty = entire workspace |
| `SWARM_MIN_OBJECTIVE_LEN` | `10` | Minimum issue title length |
| `SWARM_CLOUD_HTTP_TIMEOUT_SECS` | `300` | Per-request HTTP timeout for cloud API calls (5 min) |
| `SWARM_LOCAL_HTTP_TIMEOUT_SECS` | `2700` | Per-request HTTP timeout for local LLM calls (45 min) |
| `SWARM_BEADS_BIN` | `bd` | Beads CLI binary name |
| `SWARM_OTEL_ENDPOINT` | *(empty)* | OTLP endpoint for beads metrics (e.g., `http://victoriametrics:4318`) |
| `RUST_LOG` | `info` | Log level (see Debug & Monitoring) |

### Parallelism & Dispatch

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_PARALLEL_ISSUES` | `3` | Concurrent issues (one per node via round-robin) |
| `SWARM_CONCURRENT_SUBTASKS` | `true` | Decompose multi-file issues into parallel subtasks |

### Cost & Context Management

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_MAX_COST_PER_ISSUE` | `0.0` | Max USD per issue (0 = disabled). Cloud ~$15/M input + $75/M output |
| `SWARM_PRUNE_AFTER_ITERATION` | `3` | After N iterations, prune prompt to last 2 results + verifier output |

### Stack Profiles & Model Routing

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_STACK_PROFILE` | `hybrid_balanced_v1` | Role→model routing. Options: `hybrid_balanced_v1`, `small_specialist_v1`, `strategist_hybrid_v1` |

Stack profiles control which model handles each `SwarmRole` (Scout, Reviewer, RustWorker, GeneralWorker, Planner, Fixer, ReasoningWorker, Strategist, LocalManagerFallback, Council).

### Multi-Repo & Adapter

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_REPO_ID` | *(none)* | Repository identifier (e.g., "rust-daq", "CF-LIBS") for adapter selection |
| `SWARM_ADAPTER_ID` | *(none)* | QLoRA/LoRA adapter identifier for the coder model |

### TensorZero Feedback Loop

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_TENSORZERO_URL` | *(none)* | TZ gateway URL (e.g., `http://localhost:3000`). Routes inference through TZ for experiment tracking |
| `SWARM_TENSORZERO_PG_URL` | auto-detected | TZ Postgres URL for reading performance insights |
| `SWARM_TZ_INSIGHTS_TTL_SECS` | `1800` | Cache TTL for TZ insights (30 min) |

## Dogfood Operations

### Single Run (proof-of-life)

```bash
ssh brian@100.105.113.58 "cd ~/code/beefcake-swarm && \
  SWARM_CLOUD_API_KEY=\$SWARM_CLOUD_API_KEY \
  SWARM_CLOUD_URL=http://localhost:8317/v1 \
  SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 \
  timeout 120 bash scripts/run-swarm.sh --issue test-probe --objective 'Reply with OK'"
```

### Continuous Loop

```bash
ssh brian@100.105.113.58 "cd ~/code/beefcake-swarm && \
  nohup bash -c 'export SWARM_CLOUD_API_KEY=\$SWARM_CLOUD_API_KEY \
    SWARM_CLOUD_URL=http://localhost:8317/v1 \
    SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 \
    RUST_LOG=debug,hyper=info,reqwest=info,h2=info,rustls=info,tower=info && \
  ./scripts/dogfood-loop.sh --issue-list \"<space-separated-ids>\" --cooldown 120' \
  > ~/dogfood-debug-\$(date +%Y%m%d-%H%M).log 2>&1 &"
```

**dogfood-loop.sh options:**
- `--issue-list "id1 id2 ..."` — issues to process in order
- `--parallel N` — run N issues concurrently in batches (default: 1 = serial)
- `--cooldown N` — seconds between runs/batches (default: 60)
- `--max-runs N` — stop after N total runs (default: 0 = unlimited)
- `--discover` — auto-fetch new issues from `bd ready` when issue list exhausted
- `--repo-root <path>` — target an external repo (worktrees created in that repo)
- `--max-issue-failures N` — defer issue after N consecutive failures (default: 3)
- `DOGFOOD_LOG_DIR=./logs/dogfood` — per-run log directory

**Multi-repo support:** Use `--repo-root` to run the swarm against a different repository. Per-repo lockfiles (`/tmp/dogfood-loop-<repo-name>.lock`) prevent overlapping loops on the same target.

**API key:** `SWARM_CLOUD_API_KEY` must be set in the environment or `~/.bashrc` on ai-proxy. Never hardcode credentials in commands or documentation.

**Worktrees:** Created at `/tmp/beefcake-wt/<issue-id>`. Clean stale worktrees:

```bash
rm -rf /tmp/beefcake-wt/<issue-id> && git worktree prune
```

## Debug & Monitoring

### RUST_LOG Pattern

```bash
# Production (default)
RUST_LOG=info

# Debug with HTTP noise suppressed
RUST_LOG=debug,hyper=info,reqwest=info,h2=info,rustls=info,tower=info
```

### Monitoring Commands

```bash
# Live loop output
tail -f ~/dogfood-debug-*.log

# Per-run log
tail -f ~/code/beefcake-swarm/logs/dogfood/run-N-<issue>-*.log

# Tool call distribution (requires RUST_LOG=debug)
grep -o 'gen_ai.tool.name[^"]*"[^"]*"' logs/dogfood/run-*.log | sort | uniq -c | sort -rn

# Check endpoint health
curl -s http://vasp-03:8081/health  # Scout (Qwen3-Coder-Next)
curl -s http://vasp-01:8081/health  # Coder (122B-A10B MoE)
curl -s http://vasp-02:8081/health  # Reasoning (122B-A10B MoE)
```

### Healthy Startup Log

```text
INFO swarm_agents: Endpoint health check local_ok=true coder_ok=true reasoning_ok=true
INFO swarm_agents: Beads-free mode: processing CLI issue id=<issue>
INFO swarm_agents::agents: Building cloud-backed manager with proxy-prefixed workers model=claude-opus-4-6
```

## External Tools (install separately)

- `bd` (beads): `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash` — Go binary, issue tracker CLI. Invoked via subprocess (see `beads_bridge.rs`). Humans may also use `bdh` wrapper for convenience.
- `bv` (beads_viewer): `go install github.com/Dicklesworthstone/beads_viewer@latest`
- `gastown`: `go install github.com/steveyegge/gastown@latest` — Multi-agent workspace manager (evaluated, not adopted — swarm uses raw `git worktree` via `worktree_bridge.rs`).
- `stringer`: `go install github.com/davetashner/stringer/cmd/stringer@latest` — Codebase archaeology scanner. Outputs beads JSONL for auto-populating backlog.
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

## Environment Variables (Coordination)

**Coordination MCP:**
- `ROUTER_URL` — LLM endpoint for coordination's internal router (default: points to legacy IP, override to active endpoint)
- `ARCHITECT_MODEL`, `CODER_MODEL`, `HYDRA_MODEL` — model name overrides for coordination's router (legacy names; active swarm uses `SWARM_*` env vars instead)
- `HARNESS_MAX_ITERATIONS`, `HARNESS_FEATURES_PATH`, `HARNESS_PROGRESS_PATH`

> **Note:** The coordination MCP's `ROUTER_URL` and model env vars are a legacy interface from before swarm-agents existed. The active orchestrator uses `SWARM_*` env vars (see above). Coordination env vars only matter when running `cargo run -p coordination` standalone.

**SLURM:**
- `SLURM_SCRIPTS_PATH` (default: `/cluster/shared/scripts/llama-cpp`)
- `SLURM_ENDPOINTS_PATH` (default: `/cluster/shared/ai/endpoints`)
- `SLURM_HOST` (default: `slurm-ctl`)

**Council (cloud escalation):**
- `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

## Cluster Access

- slurm-ctl: `ssh root@10.0.0.5` (controller, NFS server — VM 500 on pve1)
- vasp-01: `ssh root@10.0.0.20` (V100S + 256GB RAM, Qwen3.5-122B-A10B MoE — VM 600 on pve1)
- vasp-02: `ssh root@10.0.0.21` (V100S + 256GB RAM, Qwen3.5-122B-A10B MoE — VM 601 on pve2)
- vasp-03: `ssh root@10.0.0.22` (V100S + 256GB RAM, Qwen3-Coder-Next 80B/3B MoE — VM 602 on pve3)
- pve1: `ssh root@10.0.0.1` (Proxmox host, cluster gateway — DO NOT reboot)
- pve2: `ssh root@10.0.0.2` (Proxmox host)
- pve3: `ssh root@10.0.0.3` (Proxmox host)
- ai-proxy: `ssh brian@100.105.113.58` or `ssh root@100.105.113.58` (LXC on pve3)
  - Codebases live under `/home/brian/code/` (beefcake-swarm, rust-daq)
  - Use `brian` user for code work; `root` for system admin only
  - GitHub auth: SSH key (`ai-proxy-lxc`) + `gh` CLI as TheFermiSea
  - CLIAPIProxy on port 8317; API key set via `SWARM_CLOUD_API_KEY` env var

## SLURM Rules

**ALL computational tasks MUST go through SLURM.** Never run workloads directly on compute nodes.

## NFS Layout

```text
/cluster/shared/
├── llama-cpp/bin/        # Inference binaries (llama-server-mmq)
├── scripts/llama-cpp/    # SLURM job scripts
├── ai/endpoints/         # Service discovery JSON
└── ai/logs/              # Shared logs
```

Worktrees are created locally at `/tmp/beefcake-wt/<issue-id>` (not on NFS). Beads data syncs via Dolt remote on ai-proxy.

## Known Issues

- `#![allow(dead_code)]` in `coordination/src/main.rs` — rmcp `#[tool_router]` macro triggers false positives. Targeted `#[allow]` used elsewhere after refactor (PR #20).
- `CLIAPIProxy ownership check` — Reports `owned_by=antigravity`. run-swarm.sh now accepts both "anthropic" and "antigravity", so ownership check passes normally.
- `vasp-03 NFS` — /home, /cluster/shared still NFS-mounted from slurm-ctl. Set `HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache` before running anything that writes to $HOME.

## Agent Teams

Enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in `.claude/settings.json`.

### Team Structure

- **Lead (Opus 4.6)**: Picks beads issues, assigns to teammates, reviews results
- **Teammates (Sonnet 4.6)**: Each works on one beads issue on a separate branch

### Teammate Workflow

1. Claim issue: `bd update <id> --status in_progress`
2. Create branch: `git checkout -b swarm/<issue-id>`
3. Implement the fix/feature
4. Quality gates auto-run on task completion (fmt, clippy, check, test)
5. Commit with conventional format and push branch

### Dogfood on ai-proxy

The swarm runs on ai-proxy (`brian@100.105.113.58`). Required env vars in `~/.bashrc`:

```bash
export SWARM_CLOUD_API_KEY="<your-api-key>"
export SWARM_CLOUD_URL="http://localhost:8317/v1"
export SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0
```

### Branch Strategy

Each teammate works on `swarm/<issue-id>`. Lead assigns non-overlapping issues.

## Beads Coordination (Native)

The swarm orchestrator uses native `bd` (beads) directly for issue tracking and coordination.
`bdh` (including `:aweb` for mail/chat) remains available for human and Claude Code agent use.

### Start Here (Every Session)

```bash
bd ready       # find unblocked work
bd list        # see all issues
```

### Rules

- Use `bd` (not `bdh`) in the swarm codebase — the orchestrator calls `bd` directly
- Humans may use `bdh` for convenience (it wraps `bd`)
- Identity is set via `BD_ACTOR` env var (e.g., `BD_ACTOR=worker-vasp03-beefcake-abc1`)
- Sync with `bd dolt push/pull` to the Dolt remote on ai-proxy

### Native Messaging (Phase 2)

```bash
BD_ACTOR=worker-1 bd mail send lead -s "Stuck on issue" -m "Details..."
bd mail inbox          # Check for messages
bd mail read <id>      # Read a specific message
```

### Event Hooks (Phase 4)

Shell scripts in `.beads/hooks/` run on issue lifecycle events. Each receives JSON issue data on stdin from beads. All hooks fail silently and never block.

| Hook | Trigger | Action |
|------|---------|--------|
| `on_close` | Issue closed | Logs `issue.closed` to `.swarm-hook-events.jsonl` |
| `on_create` | Issue created | Logs `issue.created`; auto-claims if `swarm-ready` label present |
| `on_update` | Issue updated | Logs `issue.updated` with status and actor |

Events are appended as JSONL to `$REPO_ROOT/.swarm-hook-events.jsonl` (gitignored). The orchestrator can poll this file for real-time visibility into issue state changes.

Hooks require `jq` for structured JSON output but fall back to sed-based extraction when unavailable.

### OpenTelemetry (Phase 5)

Native beads exports operational metrics via OTLP when configured:
- `bd_issue_*`: open/closed/in_progress counts (backlog health)
- `bd_storage_*`: Dolt operation duration
- Set `SWARM_OTEL_ENDPOINT=http://collector:4318` in env to enable