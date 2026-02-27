# beefcake-swarm

Autonomous coding swarm: local LLM agents on an HPC cluster (3x V100S GPUs) coordinated through deterministic quality gates.

## Architecture

```
Rig agents → Gastown worktrees → Beads tracking → SLURM dispatch
```

### Escalation Ladder

```
Cloud Manager (Opus 4.6 via CLIAPIProxy)
    → delegates to local workers: Qwen3.5-397B (vasp-01, vasp-02)
    → runs verifier after each worker completes
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

### Workspace Crates

| Crate | Description |
|-------|-------------|
| `coordination/` | Deterministic logic layer (~21k LOC). NO LLM calls, pure state machines. MCP server (rmcp) exposing tools via stdio. Includes verifier pipeline, escalation routing, ensemble voting, council escalation, harness session management, SLURM lifecycle, and OpenTelemetry-compatible tracing. |
| `crates/swarm-agents/` | Rig-based orchestrator for the agent loop. Picks beads issues, creates worktrees, runs implementer/verifier/validator cycle, merges on pass. |

### Other Directories

- `flywheel/` -- Forked TypeScript/Node project for mining prompts and task decomposition strategies.
- `indexing/` -- Python scripts for code indexing (CocoIndex for semantic search/RAG).
- `inference/` -- SLURM job scripts, systemd daemon, build/validate scripts for llama.cpp inference.
- `infrastructure/` -- GPU dashboard, HPC watchdog, ai-proxy setup.
- `docs/` -- Architecture docs, deployment guides, inference endpoint specs.

## Quick Start

```bash
cargo build --workspace                    # Build all crates
cargo test -p coordination                 # Run coordination tests
cargo test -p swarm-agents                 # Run swarm-agents tests
cargo fmt --all -- --check                 # Check formatting
cargo clippy --workspace -- -D warnings    # Lint
```

### Run

```bash
cargo run -p swarm-agents                  # Run orchestrator (needs inference running)
cargo run -p coordination                  # Run MCP server (stdio transport)
cargo run -p coordination -- --harness     # MCP server with harness tools
```

### Docker

```bash
docker compose up swarm-agents                          # Cloud-only mode
docker compose --profile local-inference up             # With local llama-server
```

## External Tools

- [`bd` (beads)](https://github.com/steveyegge/beads) -- Issue tracker CLI
- [`gastown`](https://github.com/steveyegge/gastown) -- Git worktree isolation per agent task
- [`nlm` (notebooklm-mcp-cli)](https://pypi.org/project/notebooklm-mcp-cli/) -- NotebookLM CLI for knowledge base queries

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ROUTER_URL` | LLM endpoint for coordination MCP |
| `ARCHITECT_MODEL`, `CODER_MODEL`, `HYDRA_MODEL` | Model name overrides |
| `SLURM_SCRIPTS_PATH`, `SLURM_ENDPOINTS_PATH`, `SLURM_HOST` | SLURM configuration |
| `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Cloud escalation (council) |
| `SWARM_CLOUD_URL`, `SWARM_CLOUD_API_KEY`, `SWARM_CLOUD_MODEL` | Cloud proxy configuration |
| `SWARM_ISSUE` | Target a specific beads issue by ID |

See [CLAUDE.md](CLAUDE.md) for the full list and agent contributor guidelines.

## Documentation

- [CLAUDE.md](CLAUDE.md) -- Agent instructions, model routing, cluster access, full environment reference
- [AGENTS.md](AGENTS.md) -- Issue tracking workflow, Rig patterns, session completion checklist
- [docs/architecture/](docs/architecture/) -- Deployment guides, strategy docs, architecture deep dives
