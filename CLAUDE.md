# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Selection & Routing

- **Local RPC Distributed Ensemble (March 2026):**
  - **Scout/Reviewer:** Qwen3.5-27B-Distilled (vasp-03:8081) — 100% VRAM-resident, 192K context, blazing fast.
  - **Integrator:** Qwen3.5-122B-A10B MoE (vasp-01:8081 RPC head + vasp-02 RPC worker) — 128K context, distributed VRAM.
- **General/Complex:** Claude Opus 4.6 / Sonnet 4.6 (default).
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

**`crates/swarm-agents/`** — Rig-based orchestrator (Phase 2: cloud manager + local workers). The active loop:
1. Pick highest-priority beads issue (or CLI `--issue`)
2. Create Gastown worktree in `/tmp/beefcake-wt/<issue-id>`
3. Cloud manager (Claude Opus 4.6 via CLIAPIProxy) plans and delegates
4. Local workers (Qwen3.5-27B on vasp-03, Qwen3.5-122B on vasp-01+02 RPC) execute code changes
5. Verifier (deterministic quality gates) after each iteration
6. Pass → merge + close issue; Fail → retry up to `SWARM_MAX_RETRIES`

### Escalation Ladder

```text
Cloud Manager (Claude Opus 4.6 thinking via CLIAPIProxy, max 10 iterations)
    → delegates to local workers (2026 VRAM-Resident RPC):
        vasp-03:8081 — Scout/Reviewer tier (Qwen3.5-27B-Distilled, 192K context, VRAM-resident)
        vasp-01:8081 — Integrator tier (Qwen3.5-122B-A10B MoE, 128K context, vasp-02 as RPC worker)
    → runs verifier after each worker completes
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

Cloud models are managers from iteration 1. Local models are workers.
When cloud is unavailable, falls back to worker-first mode (local models only).

### Non-Workspace Directories

- `flywheel/` — Forked from `Dicklesworthstone/agentic_coding_flywheel_setup`. TypeScript/Node. Mining prompts and task decomposition strategies; discarding Docker/cloud, adapting to SLURM/NFS.
- `indexing/` — Python scripts for code indexing (CocoIndex for semantic search/RAG).
- `inference/` — SLURM job scripts, systemd daemon, build/validate scripts.
- `infrastructure/` — Monitoring: GPU dashboard, HPC watchdog, ai-proxy setup, cloud-proxy.service (socat relay).
- `docs/` — Architecture docs, deployment guides, inference endpoint specs.

## Inference Endpoints

The swarm uses a **VRAM-Resident Distributed** architecture (Transitioning March 2026).

| Tier | Model | Node | Context | Hardware Strategy |
|------|-------|------|---------|-------------------|
| Scout/Reviewer | Qwen3.5-27B-Distilled | vasp-03 | 192K | 100% VRAM-resident (32GB) |
| Integrator | Qwen3.5-122B-A10B MoE | vasp-01 (head) + vasp-02 (RPC worker) | 128K | Dual-node RPC (64GB VRAM) |
| Cloud | Claude Opus 4.6 | ai-proxy | 200K | CLIAPIProxy Relay |

**Health Checks:**

```bash
curl -s http://vasp-03:8081/health  # Scout
curl -s http://vasp-01:8081/health  # Integrator (Head)
```

**Start inference (New RPC Ensemble):**

```bash
ssh root@10.0.0.22 "sbatch /cluster/shared/scripts/run-27b-256k.slurm"
ssh root@10.0.0.20 "sbatch /cluster/shared/scripts/run-122b-rpc.slurm"
```

**Cloud proxy:** CLIAPIProxy runs on ai-proxy (localhost:8317). Uses `x-api-key` header (not Bearer). API key set via `SWARM_CLOUD_API_KEY` env var.

**llama.cpp build:** v8231 (c024d8590), native on vasp-03 (Rocky 8.8/GCC 13/CUDA 12.6). Binary: `/usr/local/bin/llama-server-mmq` (compiled with `GGML_CUDA_FORCE_MMQ=ON` for V100), deployed to all 3 nodes. Includes autoparser refactor for Qwen3-Coder XML tool call parsing. NFS backup: `/cluster/shared/llama-cpp/bin/autoparser-build/`. Rollback: `/usr/local/bin/llama-server-mmq.b8179`.

## Swarm Environment Variables

Set by `scripts/run-swarm.sh` (overrides config.rs defaults). See `crates/swarm-agents/src/config.rs` for all options.

### Tier Endpoints

Tiers use task-specialized models (March 2026 upgrade).

| Variable | Default | Model |
|----------|---------|-------|
| `SWARM_FAST_URL` / `SWARM_FAST_MODEL` | `http://vasp-03:8081/v1` | `Qwen3.5-27B-Distilled` |
| `SWARM_CODER_URL` / `SWARM_CODER_MODEL` | `http://vasp-01:8081/v1` | `Qwen3.5-122B-A10B` |
| `SWARM_REASONING_URL` / `SWARM_REASONING_MODEL` | `http://vasp-01:8081/v1` | `Qwen3.5-122B-A10B` |

Note: vasp-02 serves as an RPC worker shard for vasp-01 and has no independent endpoint. Both `SWARM_CODER_URL` and `SWARM_REASONING_URL` point to vasp-01.

### Cloud Endpoint

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_CLOUD_URL` | `http://localhost:8317/v1` (script) / *(none)* (config.rs) | Required for cloud manager mode |
| `SWARM_CLOUD_API_KEY` | *(none)* | Required if cloud URL set |
| `SWARM_CLOUD_MODEL` | `claude-opus-4-6` | Primary cloud model |
| `SWARM_CLOUD_FALLBACK_MODEL` (script) / `SWARM_CLOUD_FALLBACK_MODELS` (config.rs) | `claude-sonnet-4-5-20250929` (script) / `claude-sonnet-4-5-20250929, gemini-2.5-flash` (config.rs) | Note singular vs plural env var name |
| `SWARM_REQUIRE_ANTHROPIC_OWNERSHIP` | `1` | run-swarm.sh accepts both "anthropic" and "antigravity" |
| `SWARM_CLOUD_PREFLIGHT` | `1` | Probe cloud endpoint before starting |

### Behavior

| Variable | Default | Notes |
|----------|---------|-------|
| `SWARM_MAX_RETRIES` | `10` | Max iterations per issue |
| `SWARM_CLOUD_MAX_RETRIES` | `3` | Cloud-specific retry limit |
| `SWARM_MAX_NO_CHANGE` | `2` | Circuit breaker for stuck iterations |
| `SWARM_CLOUD_ONLY` | `false` | Route all work through cloud (skip local) |
| `SWARM_VERIFIER_PACKAGES` | *(empty)* | Comma-separated; empty = entire workspace |
| `SWARM_MIN_OBJECTIVE_LEN` | `10` | Minimum issue title length |
| `SWARM_CLOUD_HTTP_TIMEOUT_SECS` | `300` | Per-request HTTP timeout for cloud API calls (5 min) |
| `SWARM_LOCAL_HTTP_TIMEOUT_SECS` | `900` | Per-request HTTP timeout for local LLM calls (15 min) |
| `SWARM_BEADS_BIN` | `bd` | Beads CLI binary name (must be `bd`, not `bdh` — BeadsBridge parses `bd show --json` format) |
| `RUST_LOG` | `info` | Log level (see Debug & Monitoring) |

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
- `DOGFOOD_LOG_DIR=./logs/dogfood` — per-run log directory

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
curl -s http://vasp-03:8081/health  # Scout (Qwen3.5-27B-Distilled)
curl -s http://vasp-01:8081/health  # Integrator head (Qwen3.5-122B-A10B MoE)
# vasp-02 is an RPC worker shard for vasp-01 — no independent endpoint
```

### Healthy Startup Log

```text
INFO swarm_agents: Endpoint health check local_ok=true coder_ok=true reasoning_ok=true
INFO swarm_agents: Beads-free mode: processing CLI issue id=<issue>
INFO swarm_agents::agents: Building cloud-backed manager with proxy-prefixed workers model=claude-opus-4-6
```

## External Tools (install separately)

- `bdh` (beads): `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash` — Go binary, issue tracker CLI. Invoked via subprocess (see `beads_bridge.rs`).
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

## Environment Variables (Coordination)

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
- vasp-01: `ssh root@10.0.0.20` (V100S + 256GB RAM, Qwen3.5-122B-A10B MoE RPC head — VM 600 on pve1)
- vasp-02: `ssh root@10.0.0.21` (V100S + 256GB RAM, Qwen3.5-122B-A10B MoE RPC worker — VM 601 on pve2)
- vasp-03: `ssh root@10.0.0.22` (V100S + 256GB RAM, Qwen3.5-27B-Distilled running — VM 602 on pve3)
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
├── llama-cpp/bin/        # Inference binaries
├── scripts/llama-cpp/    # SLURM job scripts
├── ai/endpoints/         # Service discovery JSON
├── ai/logs/              # Shared logs
└── (future)
    ├── gastown-town/     # Gastown workspace root
    └── beads-db/         # Shared beads database
```

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

1. Claim issue: `bdh update <id> --status in_progress`
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

<!-- BEADHUB:START -->
## BeadHub Coordination Rules

This project uses `bdh` for multi-agent coordination and issue tracking, `bdh` is a wrapper on top of `bd` (beads). Commands starting with : like `bdh :status` are managed by `bdh`. Other commands are sent to `bd`.

You are expected to work and coordinate with a team of agents. ALWAYS prioritize the team vs your particular task.

You will see notifications telling you that other agents have written mails or chat messages, or are waiting for you. NEVER ignore notifications. It is rude towards your fellow agents. Do not be rude.

Your goal is for the team to succeed in the shared project.

The active project policy as well as the expected behaviour associated to your role is shown via `bdh :policy`.

## Start Here (Every Session)

```bash
bdh :policy    # READ CAREFULLY and follow diligently
bdh :status    # who am I? (alias/workspace/role) + team status
bdh ready      # find unblocked work
```

Use `bdh :help` for bdh-specific help.

## Rules

- Always use `bdh` (not `bd`) so work is coordinated
- Default to mail (`bdh :aweb mail list|open|send`) for coordination; use chat (`bdh :aweb chat pending|open|send-and-wait|send-and-leave|history|extend-wait`) when you need a conversation with another agent.
- Respond immediately to WAITING notifications — someone is blocked.
- Notifications are for YOU, the agent, not for the human.
- Don't overwrite the work of other agents without coordinating first.
- ALWAYS check what other agents are working on with bdh :status which will tell you which beads they have claimed and what files they are working on (reservations).
- `bdh` derives your identity from the `.beadhub` file in the current worktree. If you run it from another directory you will be impersonating another agent, do not do that.
- Prioritize good communication — your goal is for the team to succeed

## Using mail

Mail is fire-and-forget — use it for status updates, handoffs, and non-blocking questions.

```bash
bdh :aweb mail send <alias> "message"                         # Send a message
bdh :aweb mail send <alias> "message" --subject "API design"  # With subject
bdh :aweb mail list                                           # Check your inbox
bdh :aweb mail open <alias>                                   # Read & acknowledge
```

## Using chat

Chat sessions are persistent per participant pair. Use `--start-conversation` when initiating a new exchange (longer wait timeout).

**Starting a conversation:**
```bash
bdh :aweb chat send-and-wait <alias> "question" --start-conversation
```

**Replying (when someone is waiting for you):**
```bash
bdh :aweb chat send-and-wait <alias> "response"
```

**Final reply (you don't need their answer):**
```bash
bdh :aweb chat send-and-leave <alias> "thanks, got it"
```

**Other commands:**
```bash
bdh :aweb chat pending          # List conversations with unread messages
bdh :aweb chat open <alias>     # Read unread messages
bdh :aweb chat history <alias>  # Full conversation history
bdh :aweb chat extend-wait <alias> "need more time"  # Ask for patience
```
<!-- BEADHUB:END -->