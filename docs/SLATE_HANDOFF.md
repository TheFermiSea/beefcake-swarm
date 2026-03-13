# Slate Architecture Upgrade — Agent Handoff

**Date**: 2026-03-13
**Epic**: `beefcake-mmn3` (Slate Architecture Upgrade: LLM-as-OS-Kernel pattern)
**Status**: Phase 0 complete, Phase 3 complete, Phases 1/2/4 open

---

## What Is Slate?

Slate treats the LLM as an **OS Kernel**: context window = RAM, agents = Processes,
sub-tasks = Threads. The goal is to eliminate context rot (raw tool output flooding
the orchestrator's context) and enable massive parallelism (multiple workers on
non-overlapping files simultaneously).

Design docs: `GEMINI_SUGGESTIONS/PHASE1.md` through `PHASE4.md`
Plan file: `~/.claude/plans/effervescent-toasting-scone.md`

---

## Completed Work

### Phase 3 — Strategy/Tactics Tool Segregation (2 issues, both closed)

| Issue | Title | Status |
|-------|-------|--------|
| `beefcake-g8rh` | Audit and categorize all swarm tools into strategy vs tactics bundles | Closed |
| `beefcake-4qzd` | Enforce strict tool segregation in cloud manager and worker agent builders | Closed |

Cloud manager (kernel) gets strategy tools only; workers (processes) get tactical tools only.
Files: `tools/bundles.rs`, `agents/cloud.rs`, `agents/coder.rs`, `agents/manager.rs`.

### Phase 0 — Orchestrator Refactor & Send-Safety (3 issues, all closed)

| Issue | Title | Commit | Status |
|-------|-------|--------|--------|
| `beefcake-fzr5` | Split orchestrator.rs into submodules | `0b33a5a` | Closed |
| `beefcake-frxr` | Refactor process_issue for Send-safety | `4395de5` | Closed |
| `beefcake-dj8a` | Integration tests for orchestrator refactor | `bd99766` | Closed |

**Phase 0a** split the monolithic `orchestrator.rs` (4454 LOC) into:
- `orchestrator/mod.rs` (3003 LOC) — lifecycle: `process_issue`, session resume, retry loop
- `orchestrator/dispatch.rs` (570 LOC) — `route_to_coder`, `format_task_prompt`, `format_compact_task_prompt`
- `orchestrator/validation.rs` (580 LOC) — `cloud_validate`, `local_validate`, feedback extraction
- `orchestrator/helpers.rs` (477 LOC) — env parsing, KB failsafe, scaffolding, directives, bdh glue

All items re-exported from `mod.rs` for backwards compatibility.

**Phase 0b** made `process_issue` Send-safe:
- Split into `process_issue` (wrapper) + `process_issue_core` (body)
- Uses `tracing::Instrument` instead of `Span::enter()` for span management
- Extracted `cluster_health.summary().await` before `info!` macro (avoids `&dyn Value` across await)
- Removed `iter_span.enter()` guard in iteration loop
- Compile-time `Send` assertion: `_assert_process_issue_core_is_send` in `mod.rs`

**Phase 0c** added 15 integration tests in `tests/orchestrator_refactor_test.rs`.

### Other Agent's Work (same session)

Commit `0199cf6` added stack profiles and strategist tier:
- `SwarmStackProfile` enum and `SwarmRole` for model-role mapping
- `SwarmTier::Strategist` (read-only advisor tier)
- Extended `MetricsCollector` and `SessionMetrics` with new fields
- Benchmark manifest (`coordination/src/benchmark/manifest.rs`)

Commit `583ce59` (ours) fixed all broken tests from the above.

---

## Remaining Work — Next Phases

### Dependency Graph

```
Phase 1 (Episodes) ──→ Phase 2 (Thread Weaving) ──→ Phase 4 (IPC)
```

Phase 1 defines types that Phase 2 and Phase 4 depend on.

### Phase 1 — Episode Architecture (3 issues, all open)

**Goal**: Workers return compressed `EpisodeSummary` instead of raw strings, preventing context rot.

| Issue | Title | Priority | Blocked By |
|-------|-------|----------|------------|
| `beefcake-5ol5` | Define EpisodeSummary and ExecutionOutcome types in coordination memory module | P1 | — |
| `beefcake-mtku` | Implement episode summarizer loop with JSON fallback in swarm-agents modes | P1 | `beefcake-5ol5` |
| `beefcake-aruf` | Wire episode summarizer into orchestrator worker return path | P1 | `beefcake-5ol5` |

**Start here**: `beefcake-5ol5` — it unblocks everything else.

**Implementation guidance** (from `GEMINI_SUGGESTIONS/PHASE1.md`):

1. Create types in `coordination/src/memory/` (new `types.rs` or add to `store.rs`):
   ```rust
   pub struct EpisodeSummary {
       pub thread_id: String,
       pub task_objective: String,
       pub tactical_actions_taken: Vec<String>,
       pub outcome: ExecutionOutcome,
       pub discovered_knowledge: Option<String>,
       pub files_modified: Vec<String>,
       pub iteration_count: u32,
       pub token_usage: Option<u64>,
   }

   pub enum ExecutionOutcome {
       Success(String),       // summary of what was accomplished
       Blocked(String),       // requires kernel intervention
       Failed { error: String, retryable: bool },
   }
   ```

2. Summarizer loop (`beefcake-mtku`): After worker completes, a summarizer agent compresses
   the raw trace into `EpisodeSummary` JSON. JSON fallback: if LLM fails, truncate raw trace
   to last 50 lines wrapped in a `Failed` outcome. Existing `Summarizer` trait in
   `coordination/src/memory/summarizer.rs` can be reused.

3. Wire into orchestrator (`beefcake-aruf`): Replace raw string returns from workers with
   `EpisodeSummary`. The orchestrator kernel stores only summaries, not full traces.

### Phase 2 — Thread Weaving (4 issues, all open)

**Goal**: Replace serial subtask dispatch with concurrent `tokio::spawn` into isolated worktrees.

| Issue | Title | Priority | Blocked By |
|-------|-------|----------|------------|
| `beefcake-p9gc` | Implement ThreadWeaver struct with mpsc channels and Gastown worktree dispatch | P1 | Phase 1 types |
| `beefcake-32xq` | Add dispatch_thread tool for cloud manager to replace serial delegate_task | P1 | `beefcake-p9gc` |
| `beefcake-d8cy` | Add RocksDB checkpoint persistence for active thread recovery | P2 | `beefcake-p9gc` |
| `beefcake-wotk` | Integration test: concurrent workers on non-overlapping files merge successfully | P2 | `beefcake-p9gc` |

**Key**: `process_issue_core` is now `Send` (Phase 0b), so `tokio::spawn` / `JoinSet` will compile.
Currently `dispatch_parallel_issues` in `main.rs` still uses `std::thread::spawn + Handle::block_on`
(the old `!Send` workaround) — Phase 2 replaces this with proper async dispatch.

### Phase 4 — BeadHub IPC (3 issues, all open)

**Goal**: Use bdh mail/chat as inter-process communication between kernel and threads.

| Issue | Title | Priority | Blocked By |
|-------|-------|----------|------------|
| `beefcake-snqx` | Add yield_episode wrapper to BdhBridge for thread-to-kernel episode return | P1 | `beefcake-5ol5` |
| `beefcake-1vll` | Add sys_interrupt wrapper to BdhBridge for fatal error escalation via chat | P1 | `beefcake-5ol5` |
| `beefcake-vd8c` | Update orchestrator event loop to poll bdh mail for completed episodes | P1 | Phase 2 |

### Cross-Cutting (1 issue)

| Issue | Title | Priority |
|-------|-------|----------|
| `beefcake-m3cz` | End-to-end validation: full Slate pipeline from dispatch through episode return | P2 |

---

## Hazards & Gotchas

### Multi-Agent Contention

Another agent may be working on this repo simultaneously. **Always use `git worktree`**
for any multi-file changes to avoid dirty-tree conflicts:

```bash
git worktree add /tmp/beefcake-wt/<branch-name> -b <branch-name> HEAD
# ... work in worktree ...
git merge <branch-name> --ff-only  # from main worktree
git worktree remove /tmp/beefcake-wt/<branch-name>
git branch -d <branch-name>
```

### Stale Task List

The Claude Code task list (#4-#11) is frozen from a prior session and cannot be updated.
Ignore it. Use **beads issues** (`bd show`, `bd ready`, `bd close`) as the source of truth.

### Git Stash

There are 9 stash entries, mostly from prior sessions. `stash@{0}` is from the Phase 0a
commit. These are historical WIP and can be ignored unless you're specifically asked to
recover something.

### ai-proxy Deployment

- SSH: `brian@100.105.113.58`
- Code: `~/code/beefcake-swarm`
- Deploy: `git fetch origin && git reset --hard origin/main && cargo check -p swarm-agents`
- API key: `export SWARM_CLOUD_API_KEY=rust-daq-proxy-key` (not in .bashrc for non-interactive shells)
- Stale worktrees: `rm -rf /tmp/beefcake-wt/<id> && git worktree prune`

### Send-Safety Invariant

Do NOT re-introduce `Span::enter()` guards held across `.await` points in
`process_issue_core`. The compile-time assertion at `mod.rs:2350` will catch this,
but be aware. Use `tracing::Instrument` for new spans.

### coordination vs swarm-agents Boundary

New **types** go in `coordination/` (pure logic, no LLM calls).
New **orchestration code** goes in `crates/swarm-agents/`.
This is a firm architectural rule — `coordination` is the deterministic layer.

---

## Recommended Next Steps

1. **Start with `beefcake-5ol5`** — Define `EpisodeSummary` and `ExecutionOutcome` in
   `coordination/src/memory/`. This is the keystone type that unblocks 4 downstream issues.

2. **Then `beefcake-mtku`** — Implement the summarizer loop. Reuse the existing `Summarizer`
   trait from `coordination/src/memory/summarizer.rs`.

3. **Then `beefcake-aruf`** — Wire it into the orchestrator return path.

4. Phase 2 and 4 can begin once Phase 1 types are defined.

---

## Quick Reference

```bash
bd show beefcake-mmn3          # Epic overview with all children
bd ready                       # What's unblocked right now
bd update <id> --status=in_progress  # Claim an issue
bd close <id> --reason="..."   # Close when done
cargo test -p swarm-agents     # 53 tests (35 unit + 15 integration + 3 verifier)
cargo test -p coordination     # 26 tests
cargo clippy --workspace -- -D warnings  # Must be clean
```
