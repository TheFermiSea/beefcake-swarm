# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bdh onboard` to get started.

## Quick Reference

```bash
bdh ready              # Find available work
bdh show <id>          # View issue details
bdh update <id> --status in_progress  # Claim work
bdh close <id>         # Complete work
bdh sync               # Sync with git
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
   bdh sync
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

# Check endpoint health (2026 RPC Ensemble)
curl -s http://vasp-03:8081/health  # Scout/Reviewer (27B, 256K)
curl -s http://vasp-01:8081/health  # Integrator (122B RPC Head)
```

### Swarm Behavior Insights

- Cloud manager does heavy read-first exploration (~70% reads) before writing
- Worker delegation uses `proxy_` prefixed tools (e.g., `proxy_rust_coder`, `proxy_reasoning_worker`)
- `MaxTurnError` on a worker is expected — manager retries with a different worker
- Verifier runs after each iteration, not just at the end

## Swarm Architecture Quick Reference

```text
┌─────────────────────────────────────────────────────┐
│  Cloud Manager (Claude Opus 4.6 thinking)           │
│  via CLIAPIProxy on ai-proxy:8317                   │
│  ────────────────────────────────────               │
│  Plans work, delegates to local workers,            │
│  reads/analyzes code, runs verifier                 │
├─────────────────────────────────────────────────────┤
│  Local Workers (2026 VRAM-Resident RPC)             │
│  ┌─────────────────┬────────────────────────────────┐ │
│  │ vasp-03:8081    │ vasp-01:8081 (Head)            │ │
│  │ Scout / Reviewer│ Integrator (RPC)               │ │
│  │ (27B, 256K CTX) │ (122B, 128K CTX)               │ │
│  └─────────────────┴────────────────────────────────┘ │
│                    │ vasp-02:50052 (Worker)         │ │
│                    │ RPC Shard (27 layers)          │ │
│                    └────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│  Verifier (deterministic quality gates)             │
│  cargo fmt → clippy → cargo check → cargo test      │
└─────────────────────────────────────────────────────┘
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
bdh ready --json
```

**Create new issues:**

```bash
bdh create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bdh create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bdh update <id> --claim --json
bdh update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bdh close bd-42 --reason "Completed" --json
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

1. **Check ready work**: `bdh ready` shows unblocked issues
2. **Claim your task atomically**: `bdh update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bdh create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bdh close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bdh dolt push`/`bdh dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bdh ready` before asking "what should I work on?"
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
   bdh sync
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