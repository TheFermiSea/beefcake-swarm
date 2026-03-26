# Beefcake Swarm Development Guide

## 1. System Overview

The beefcake-swarm runs on a 3-node HPC cluster with NVIDIA V100S GPUs:

- **3-Node Independent Instances**: vasp-03 (Scout/Fast 27B-Opus-Distilled), vasp-01 (Coder 122B MoE), vasp-02 (Reasoning 122B MoE)
- **Scheduler**: SLURM for all compute (inference jobs, future agent jobs)
- **Storage**: NFS shared from slurm-ctl (10.0.0.5)
- **Gateway**: ai-proxy LXC for external access

## 2. Component Deep Dive

### Rig (Agent Framework)

Rig provides the agent abstraction for LLM interactions. We use it with OpenAI-compatible endpoints pointing at local llama-server instances.

```rust
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;

// Create a client pointing at local llama-server
let client = openai::CompletionsClient::builder()
    .api_key("not-needed")
    .base_url("http://vasp-03:8081/v1")
    .build()?;

// Build an agent with a system prompt
let agent = client
    .agent("Qwen3.5-27B-Opus-Distilled")
    .preamble("You are an expert Rust developer.")
    .build();

// Prompt and get a response
let response: String = agent.prompt("Write a function to sort a Vec<i32>").await?;
```

**If JSON adherence is flaky**: Use llama.cpp GBNF grammar constraint via the `grammar` parameter in the API request.

### beads (Task Tracking)

Go binary (`bd`), invoked via CLI subprocess.

```bash
bd list --status=open --json    # Machine-readable output
bd create --title="..." --type=task --priority=2
bd update <id> --status=in_progress
bd close <id> --reason="Done"
```

The `beads_bridge.rs` module wraps these CLI calls.

### Git Worktree Isolation

Creates git worktrees per agent task, preventing file conflicts when multiple agents work in parallel. Uses raw `git worktree` commands via `worktree_bridge.rs` (Gastown evaluated but not adopted — runtime model mismatch with HTTP-to-GPU inference).

```bash
# Handled by worktree_bridge.rs — not called directly
git worktree add /tmp/beefcake-wt/<issue-id> -b swarm/<issue-id>
# ... agent works in worktree ...
git worktree remove /tmp/beefcake-wt/<issue-id>
```

### Coordination Crate (Deterministic Logic)

Copied from `beefcake2/tools/rust-cluster-mcp/`. No LLM calls — pure Rust state machines:

- **Verifier**: `cargo fmt` → `clippy` → `cargo check` → `cargo test` pipeline
- **Escalation**: State machine (Implementer → Integrator → Cloud → Human)
- **Ensemble**: Multi-model voting with arbitration
- **Council**: Cloud escalation for disputes
- **SLURM**: Inference endpoint management and health checks

## 3. The 2-Agent Loop (MVP)

```
                ┌──────────────────────────────────┐
                │   Cloud Manager (claude-opus-4-6) │
                │   Plans and delegates via proxy   │
                └──────┬──────────────┬────────────┘
                       │              │
                ┌──────▼──────┐ ┌─────▼──────────┐
                │   Workers   │ │   Verifier      │
                │ vasp-03:8081│ │ (deterministic)  │
                │ vasp-01:8081│ │ fmt→clippy→test  │
                │ vasp-02:8081│ │                  │
                └──────┬──────┘ └─────┬──────────┘
                       │              │
                ┌──────▼──────────────▼────────────┐
                │        Git Worktree               │
                │  /tmp/beefcake-wt/<issue_id>      │
                └──────────────────────────────────┘
```

1. **Orchestrator**: Pick highest-priority beads issue (or CLI `--issue`)
2. **Environment**: Create git worktree in `/tmp/beefcake-wt/<issue-id>`
3. **Cloud Manager**: claude-opus-4-6 plans and delegates via CLIAPIProxy
4. **Workers**: Local LLMs (27B-Opus-Distilled on vasp-03, 122B on vasp-01 + vasp-02) execute code changes
5. **Verifier**: `cargo fmt && cargo clippy && cargo check && cargo test` (deterministic)
6. **If pass**: merge + close issue
7. **If fail**: retry up to `SWARM_MAX_RETRIES`

## 4. Context Building (Critical Gap)

Agents need a "repo packer" to build context windows:

- **Options**: tree-sitter AST extraction, `repomap` (Aider-style), or custom Rust walker
- **Requirements**: Respect `.gitignore`, count tokens, fit in model context
- **Semantic search**: `indexing/index_flow_v2.py` (CocoIndex) for RAG-style retrieval

## 5. SLURM Integration

```bash
# Start inference (independent instances)
ssh root@10.0.0.22 "sbatch /cluster/shared/scripts/run-27b-256k.slurm"
ssh root@10.0.0.20 "sbatch /cluster/shared/scripts/run-122b-rpc.slurm"

# Future: agent batch jobs
sbatch --dependency=afterok:$JOB_ID agent-task.slurm
```

**Rule**: ALL compute via SLURM. Never run directly on nodes.

## 6. NFS Layout

```
/cluster/shared/
├── llama-cpp/bin/        # Inference binaries
├── scripts/llama-cpp/    # SLURM scripts
├── ai/endpoints/         # Service discovery JSON
├── ai/logs/              # Shared logs
└── (future)
    ├── worktrees/        # Agent worktrees (currently at /tmp/beefcake-wt/)
    └── beads-db/         # Shared beads database
```

## 7. Moving to ai-proxy

When ready to run on cluster:

```bash
ssh root@100.105.113.58
cd /root
git clone git@github.com:TheFermiSea/beefcake-swarm.git
cd beefcake-swarm
source ~/.cargo/env
cargo build --workspace
```
