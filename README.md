# beefcake-swarm

Autonomous coding swarm: local LLM agents on an HPC cluster (3x V100S GPUs) coordinated through deterministic quality gates.

## Architecture

```
Rig agents → git worktree isolation → Beads tracking → SLURM dispatch
```

### Escalation Ladder

```
Cloud Manager (Opus 4.6 via CLIAPIProxy)
    → cloud fallback: Opus 4.6 → Gemini 3.1 Pro → Sonnet 4.6 → Flash Lite
    → delegates to local workers:
        - Scout (Qwen3.5-27B-Opus-Distilled, 65K ctx on vasp-03)
        - Coder (Qwen3.5-122B-A10B MoE, 65K ctx on vasp-01)
        - Reasoning (Qwen3.5-122B-A10B MoE, 65K ctx on vasp-02)
    → runs verifier after each worker completes (multi-language)
    → circuit breaker after consecutive no-change iterations
    ↓ all budgets exhausted
Human Intervention (blocking beads issue)
```

### Workspace Crates

| Crate | Description |
|-------|-------------|
| `coordination/` | Deterministic logic layer (~21k LOC). NO LLM calls, pure state machines. MCP server (rmcp) exposing tools via stdio. Includes multi-language verifier pipeline, escalation routing, ensemble voting, council escalation (Gemini 3.1 Pro / Opus 4.6 / GPT-5.2 Codex), harness session management, SLURM lifecycle, and OpenTelemetry tracing. |
| `crates/swarm-agents/` | Rig-based orchestrator. Picks beads issues (up to 3 concurrently), creates worktrees, runs implement/verify cycle with cloud manager + local workers. Supports multi-language (Rust, Python, TypeScript, Go), TensorZero feedback loops, stack profiles for model routing, and mutation archiving. |

### Other Directories

- `scripts/` -- Orchestration: `run-swarm.sh`, `dogfood-loop.sh`, benchmarking, deployment, issue generation.
- `config/` -- TensorZero configuration: experiment tracking, A/B testing, evaluation functions.
- `inference/` -- SLURM job scripts, systemd daemon, build/validate scripts for llama.cpp inference.
- `infrastructure/` -- GPU dashboard, HPC watchdog, ai-proxy setup, TensorZero docker-compose.
- `indexing/` -- Python scripts for code indexing (CocoIndex for semantic search/RAG).
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

## Dogfood Loop

The primary operational mode. Continuously processes beads issues on ai-proxy:

```bash
./scripts/dogfood-loop.sh --parallel 3 --cooldown 120 --discover
```

Supports multi-repo targeting (`--repo-root <path>`), per-issue circuit breakers, and per-repo lockfiles. See [CLAUDE.md](CLAUDE.md#dogfood-operations) for full options.

## External Tools

- [`bd` (beads)](https://github.com/steveyegge/beads) -- Issue tracker CLI
- [`gastown`](https://github.com/steveyegge/gastown) -- Multi-agent workspace manager (evaluated, not adopted; swarm uses raw `git worktree`)
- [`stringer`](https://github.com/davetashner/stringer) -- Codebase archaeology scanner, outputs beads JSONL
- [`nlm` (notebooklm-mcp-cli)](https://pypi.org/project/notebooklm-mcp-cli/) -- NotebookLM CLI for knowledge base queries

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `SWARM_CLOUD_URL`, `SWARM_CLOUD_API_KEY`, `SWARM_CLOUD_MODEL` | Cloud proxy (CLIAPIProxy) configuration |
| `SWARM_FAST_MODEL`, `SWARM_CODER_MODEL`, `SWARM_REASONING_MODEL` | Local model overrides per tier |
| `SWARM_PARALLEL_ISSUES` | Concurrent issues (default: 3, one per node) |
| `SWARM_STACK_PROFILE` | Model routing profile (`hybrid_balanced_v1`, `small_specialist_v1`, `strategist_hybrid_v1`) |
| `SWARM_TENSORZERO_URL` | TensorZero gateway for experiment tracking and A/B testing |
| `SWARM_REPO_ID` | Target repository identifier for multi-repo support |
| `SWARM_MAX_RETRIES` | Max iterations per issue (default: 10) |
| `SWARM_MAX_NO_CHANGE` | Circuit breaker threshold (default: 2) |
| `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Cloud escalation (council) |

See [CLAUDE.md](CLAUDE.md) for the full list (~40 env vars) and agent contributor guidelines.

## Documentation

- [CLAUDE.md](CLAUDE.md) -- Agent instructions, model routing, cluster access, full environment reference (~40 env vars)
- [AGENTS.md](AGENTS.md) -- Issue tracking workflow, Rig patterns, session completion checklist
- [docs/inference/INFERENCE-ENDPOINTS.md](docs/inference/INFERENCE-ENDPOINTS.md) -- Authoritative inference endpoint reference
- [docs/architecture/](docs/architecture/) -- Deployment guides, strategy docs, architecture deep dives
