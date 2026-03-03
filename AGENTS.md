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

# Check endpoint health (all Qwen3.5-397B)
curl -s http://vasp-03:8081/health  # fast
curl -s http://vasp-01:8081/health  # coder
curl -s http://vasp-02:8081/health  # reasoning
```

### Swarm Behavior Insights

- Cloud manager does heavy read-first exploration (~70% reads) before writing
- Worker delegation uses `proxy_` prefixed tools (e.g., `proxy_rust_coder`, `proxy_reasoning_worker`)
- `MaxTurnError` on a worker is expected — manager retries with a different worker
- Verifier runs after each iteration, not just at the end

## Swarm Architecture Quick Reference

```text
┌─────────────────────────────────────────────────┐
│  Cloud Manager (Claude Opus 4.6 thinking)       │
│  via CLIAPIProxy on ai-proxy:8317               │
│  ─────────────────────────────────              │
│  Plans work, delegates to local workers,        │
│  reads/analyzes code, runs verifier             │
├─────────────────────────────────────────────────┤
│  Local Workers (all Qwen3.5-397B, proxy_ tools) │
│  ┌──────────────┬───────────────┬─────────────┐ │
│  │ vasp-03:8081 │ vasp-01:8081  │ vasp-02:8081│ │
│  │ Fast/Scout   │ Coder         │ Reasoning   │ │
│  │ review,break │ code gen      │ plan,analyze│ │
│  └──────────────┴───────────────┴─────────────┘ │
├─────────────────────────────────────────────────┤
│  Verifier (deterministic quality gates)         │
│  cargo fmt → clippy → cargo check → cargo test  │
└─────────────────────────────────────────────────┘
```

## Common Gotchas

| Gotcha | Solution |
|--------|----------|
| CLIAPIProxy reports `owned_by=antigravity` | run-swarm.sh now accepts "antigravity" — no workaround needed |
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

<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

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

<!-- END BEADS INTEGRATION -->
