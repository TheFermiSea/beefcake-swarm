# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## Code Conventions

### Commits

Use **conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

Branch naming: `feat/<description>`, `fix/<description>`, or `swarm/<issue-id>` for automated work.

### Quality Gates (must pass before merge)

```bash
cargo fmt --all -- --check                 # Formatting
cargo clippy --workspace -- -D warnings    # Linting (warnings = errors)
cargo check --workspace                    # Compilation
cargo test -p coordination                 # Unit tests (coordination)
cargo test -p swarm-agents                 # Unit tests (swarm-agents)
```

### Rust Style

- Use `thiserror` for error types — never `Result<(), String>`
- Use `?` operator — avoid `.unwrap()` (use `.expect("reason")` only in tests)
- `WasmCompatSend` / `WasmCompatSync` for trait bounds (Rig convention)
- Targeted `#[allow(dead_code)]` with reason, not blanket suppression
- Prefer `tracing::info!`/`debug!` over `println!` for runtime logging

## Testing Strategy

### Running Tests

```bash
cargo test -p coordination                    # All coordination tests
cargo test -p coordination -- test_name       # Single test
cargo test -p swarm-agents                    # Swarm orchestrator tests
cargo test --workspace                        # Everything (slow)
```

### What to Test

- State machines: test each transition, edge cases, error paths
- Verifier: test with real rustc output snippets (see `mod tests` in `coordination/src/verifier/pipeline.rs`, `normalized.rs`, etc.)
- Config: test env var parsing; run with `--test-threads=1` for isolation (tests mutate env)
- Tools: test JSON schema generation and parameter validation

### Known Test Patterns

- Tests that modify env vars should run serially via `cargo test -- --test-threads=1`
- Config tests directly call `set_var`/`remove_var` (no restore — rely on serial execution)
- Integration tests may need inference endpoints running — skip in CI with `#[ignore]`

## Operational Patterns

### Debug Logging

```bash
# Production (default)
RUST_LOG=info

# Debug with HTTP noise suppressed
RUST_LOG=debug,hyper=info,reqwest=info,h2=info,rustls=info,tower=info
```

### Monitoring the Dogfood Loop

```bash
# Live loop output
tail -f ~/dogfood-debug-*.log

# Per-run log
tail -f ~/code/beefcake-swarm/logs/dogfood/run-N-<issue>-*.log

# Tool call distribution (requires RUST_LOG=debug)
grep -o 'gen_ai.tool.name[^"]*"[^"]*"' logs/dogfood/run-*.log | sort | uniq -c | sort -rn

# Check endpoint health
curl -s http://vasp-03:8080/v1/models | python3 -m json.tool  # fast
curl -s http://vasp-01:8081/v1/models | python3 -m json.tool  # coder
curl -s http://vasp-02:8081/v1/models | python3 -m json.tool  # reasoning
```

### Swarm Behavior Insights

- Cloud manager does heavy read-first exploration (~70% reads) before writing
- Worker delegation uses `proxy_` prefixed tools (e.g., `proxy_rust_coder`, `proxy_reasoning_worker`)
- `MaxTurnError` on a worker is expected — manager retries with a different worker
- Verifier runs after each iteration, not just at the end

## Swarm Architecture Quick Reference

```text
┌─────────────────────────────────────────────────┐
│  Cloud Manager (Claude Sonnet 4.6)              │
│  via CLIAPIProxy on ai-proxy:8317               │
│  ─────────────────────────────────              │
│  Plans work, delegates to local workers,        │
│  reads/analyzes code, runs verifier             │
├─────────────────────────────────────────────────┤
│  Local Workers (proxy_ prefixed tools)          │
│  ┌──────────────┬───────────────┬─────────────┐ │
│  │ HydraCoder   │ Qwen3-Coder  │ Qwen3.5-397B│ │
│  │ 30B (vasp-03)│ 80B (vasp-01)│ (vasp-02)   │ │
│  │ Fast analysis│ Code gen      │ Reasoning   │ │
│  └──────────────┴───────────────┴─────────────┘ │
├─────────────────────────────────────────────────┤
│  Verifier (deterministic quality gates)         │
│  cargo fmt → clippy → cargo check → cargo test  │
└─────────────────────────────────────────────────┘
```

## Common Gotchas

| Gotcha | Solution |
|--------|----------|
| CLIAPIProxy reports `owned_by=antigravity` | Set `SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0` |
| `run-swarm.sh` eats CLI args | Fixed in PR #21 — `--` separator added |
| Stale worktree blocks new run | `rm -rf /tmp/beefcake-wt/<id> && git worktree prune` |
| `SWARM_CLOUD_URL` wrong on ai-proxy | Use `http://localhost:8317/v1`, not `http://10.0.0.5:8317/v1` |
| Tests fail with env var races | Run with `cargo test -- --test-threads=1` |
| Rig `default_max_turns` not enforced | Only wall-clock timeout works with `.prompt()` |
| `nlm` not found on ai-proxy | Expected — swarm runs without NotebookLM on ai-proxy |

## Rig Framework Reference

For Rig API documentation (agents, tools, providers, streaming), use the skill:

```bash
/rig
```

Or the full reference at https://docs.rig.rs
