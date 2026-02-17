# Beefcake Swarm Development Guide

## 1. System Overview

The beefcake-swarm runs on a 3-node HPC cluster with NVIDIA V100S GPUs:

- **2+1 Topology**: vasp-02 (fast 14B), vasp-01+vasp-03 (reasoning 72B via RPC)
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
    .base_url("http://vasp-02:8080/v1")
    .build()?;

// Build an agent with a system prompt
let agent = client
    .agent("strand-rust-coder-14b-q8_0")
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

### Gastown (Workspace Isolation)

Creates git worktrees per agent task on NFS, preventing file conflicts when multiple agents work in parallel.

```bash
gastown create <issue_id>   # Create isolated worktree
# ... agent works in worktree ...
gastown merge               # Merge back to main
```

### Coordination Crate (Deterministic Logic)

Copied from `beefcake2/tools/rust-cluster-mcp/`. No LLM calls — pure Rust state machines:

- **Verifier**: `cargo fmt` → `clippy` → `cargo check` → `cargo test` pipeline
- **Escalation**: State machine (Implementer → Integrator → Cloud → Human)
- **Ensemble**: Multi-model voting with arbitration
- **Council**: Cloud escalation for disputes
- **SLURM**: Inference endpoint management and health checks

### Flywheel (Bootstrap)

Forked from `Dicklesworthstone/agentic_coding_flywheel_setup`. Adapting for SLURM/NFS:

- **Steal**: Prompts, task decomposition strategies, tool configurations
- **Discard**: Docker Compose, cloud provider setup, VPS bootstrapping
- **Adapt**: Replace Docker/cloud assumptions with SLURM/NFS patterns

## 3. The 2-Agent Loop (MVP)

```
                ┌──────────────────────────────────┐
                │   Orchestrator (swarm-agents)     │
                │   Queries beads for next TODO     │
                └──────┬──────────────┬────────────┘
                       │              │
                ┌──────▼──────┐ ┌─────▼──────────┐
                │ Implementer │ │   Validator     │
                │ (72B tier)  │ │   (14B tier)    │
                │ vasp-01:8081│ │   vasp-02:8080  │
                └──────┬──────┘ └─────┬──────────┘
                       │              │
                ┌──────▼──────────────▼────────────┐
                │        Gastown Worktree           │
                │  /cluster/shared/wt/<issue_id>    │
                └──────────────────────────────────┘
```

1. **Orchestrator**: `br list --status=open --json` → pick highest-priority TODO
2. **Environment**: `gastown create <issue_id>` → isolated worktree
3. **Implementer**: Rig agent calls 72B, reads files, writes code
4. **Verifier**: `cargo fmt && cargo clippy && cargo test` (deterministic)
5. **Validator**: Rig agent calls 14B, reviews diff *without seeing implementer context* (blind)
6. **If pass**: `gastown merge` → `br close <id>`
7. **If fail**: `br update <id> --notes "validator feedback"` → loop back to step 3

## 4. Context Building (Critical Gap)

Agents need a "repo packer" to build context windows:

- **Options**: tree-sitter AST extraction, `repomap` (Aider-style), or custom Rust walker
- **Requirements**: Respect `.gitignore`, count tokens, fit in model context
- **Semantic search**: `indexing/index_flow_v2.py` (CocoIndex) for RAG-style retrieval

## 5. SLURM Integration

```bash
# Start inference
sbatch inference/slurm/run-14b.slurm
sbatch inference/slurm/run-72b-distributed.slurm

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
    ├── gastown-town/     # Gastown workspace root
    ├── worktrees/        # Agent worktrees
    └── beads-db/         # Shared beads database
```

## 7. Moving to ai-proxy

When ready to run on cluster:

```bash
ssh root@100.105.113.58
cd /root
git clone git@github.com:TheFermiSea/beefcake-swarm.git
cd beefcake-swarm
git submodule update --init
source ~/.cargo/env
cargo build --workspace
```
