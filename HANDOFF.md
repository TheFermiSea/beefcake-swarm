# Handoff: Parallel Worker Orchestration + Inter-Agent Communication (Phase 1-3 Complete)

**Status**: ✅ Four critical fixes deployed, tested (498 tests), compiled, ready for dogfood validation
**Date**: 2026-03-12
**Previous Session**: Implemented & deployed fixes; inference endpoints restarted

---

## What Was Done

The "Parallel Worker Orchestration + Inter-Agent Communication" epic (Issues 1-6 from plan) has been **implemented and deployed**. Four critical bugs identified in dogfood runs have been **fixed and verified**.

### Issues Completed

| Issue | Title | Status | Key File | Details |
|-------|-------|--------|----------|---------|
| #1 | Enable concurrent dispatch by default | ✅ DONE | `config.rs` line ~1774 | `concurrent_subtasks` defaults to `true`; env var `SWARM_CONCURRENT_SUBTASKS` overrides |
| #2 | Manager-guided subtask planning | ✅ DONE | `orchestrator.rs` line ~2587 | Manager calls `plan_parallel_work` tool → plan deposited in `PlanSlot` → orchestrator picks up and dispatches |
| #3 | Inter-worker awareness via workpad | ✅ DONE | `tools/workpad_tool.rs` (new, 313 LOC) | Append-only JSONL workpad; workers use `announce` + `check_announcements` tools for interface change awareness |
| #4 | Integration file handling — serial post-pass | ✅ DONE | `subtask.rs` planner prompt + `orchestrator.rs` | Planner prevents multiple subtasks from touching `Cargo.toml`, `mod.rs`, `lib.rs` via validation rules |
| #5 | Failed subtask serial retry | ✅ DONE | `orchestrator.rs` post-dispatch block | Partial success (some workers succeed, some fail) now retries only failed subtasks, preserving successful progress |
| #6 | Concurrent dispatch observability | ✅ DONE | `orchestrator.rs` logging | Structured logging at dispatch, worker completion, post-pass; metrics logged |
| #7 | Manager oversight via status tool | ⏸️ DEFERRED | N/A | Only implement if Phase 1-6 dogfooding proves insufficient (expected: not needed for 2-worker case) |

### Four Fixes Applied (Dogfood Feedback)

After running dogfood with the initial implementation, **three critical bugs were identified**. All have been **fixed and deployed**:

#### Fix 1: Subtask Write Deadline Exhaustion
**Problem**: Subtask workers hit `max_turns_without_write: 5` limit before writing any code.
**Root Cause**: Even scoped subtasks need 6-8 turns to read target_files + context_files before writing.
**Solution** (lines ~513, ~527 in `subtask.rs`):
```rust
let default_max_turns = 15;  // was 10
let max_turns_without_write = Some(8);  // was Some(5)
```
**Verification**: Subtask tests pass; write deadline sanity tests added.

#### Fix 2: All-Fail Escalation Stuck at Worker Tier
**Problem**: When all concurrent subtasks fail, sequential retry starts at Worker tier and wastes 9+ iterations before escalation fires.
**Root Cause**: Escalation state wasn't reset; escalation triggers (no-change circuit breaker) take multiple iterations to fire.
**Solution** (line ~1920 in `orchestrator.rs`):
```rust
if outcome.succeeded == 0 {
    escalation.current_tier = SwarmTier::Council;
    info!(id = %issue.id, "All subtasks failed → escalating to Council");
}
```
**Verification**: Tests confirm Council tier reached immediately on all-fail.

#### Fix 3: Manager Has_Written False-Negative
**Problem**: Cloud manager delegates to `proxy_rust_coder` / `proxy_fixer` but adapter never sees nested `edit_file` calls → `has_written=false`.
**Root Cause**: Manager's RuntimeAdapter only sees tool names, not invocation graph; missing write detection for delegated work.
**Solution** (lines ~2707-2716 in `orchestrator.rs`):
```rust
let manager_delegated = tier == SwarmTier::Council
    && git_has_changes  // Source of truth: actual git diff
    && adapter_report.total_tool_calls > 0;

if manager_delegated {
    info!(id = %issue.id, "Manager delegated work (git changes detected)");
    has_written = true;
}
```
**Verification**: Manager delegation tests pass; git-backed detection confirmed.

#### Fix 4: Single-Task Issues Waste Worker Iterations
**Problem**: When planner returns 1 subtask, system falls through to normal sequential loop at Worker tier.
**Root Cause**: Single subtasks don't benefit from concurrent dispatch; local worker budget exhausted before cloud manager is consulted.
**Solution** (line ~2066 in `orchestrator.rs`):
```rust
if plan.subtasks.len() == 1 && config.cloud_available() {
    escalation.current_tier = SwarmTier::Council;
    info!(id = %issue.id, "Single-subtask plan → escalating to Council");
}
```
**Verification**: Single-task routing tests pass; escalation ladder tests confirm Council reached.

---

## Current State

### Infrastructure
- **Inference Endpoints**: ✅ Online (as of 2026-03-12 ~11:30)
  - Scout/Fast (Qwen3.5-27B-Opus-Distilled Q4_K_M): vasp-03:8081
  - Coder (Qwen3.5-122B-A10B MoE): vasp-01:8081
  - Reasoning (Qwen3.5-122B-A10B MoE): vasp-02:8081
  - Both responding to health checks: `curl -s http://10.0.0.22:8081/health` → `{"status":"ok"}`

- **Cloud Proxy**: ✅ Running on ai-proxy (localhost:8317), CLIAPIProxy relay active

### Code
- **All fixes committed** to main branch (git log shows latest commits: `54706cd` for single-task routing, `5a9ce05` for subtask budget + all-fail)
- **Tests**: 498 passing, 0 failing (`cargo test -p swarm-agents`)
- **Build**: Clean (no clippy warnings, no compilation errors)

### Configuration
- **Concurrent dispatch**: Enabled by default (`SWARM_CONCURRENT_SUBTASKS` defaults to `true`)
- **Manager planning**: Wired in (`plan_parallel_work` tool available to cloud manager)
- **Workpad**: Initialized before each concurrent dispatch; `.swarm-workpad.jsonl` created in worktree
- **Write deadline**: 8 turns (read exploration) + 15 total turns per subtask

---

## Key Implementation Details

### Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `crates/swarm-agents/src/config.rs` | Added `concurrent_subtasks: bool` field; env var override | ~1774 |
| `crates/swarm-agents/src/orchestrator.rs` | Plan slot pickup (2587), all-fail escalation (1920), single-task routing (2066), manager delegation detection (2707) | ~4 blocks |
| `crates/swarm-agents/src/subtask.rs` | Write deadline: `max_turns_without_write: 8`, max turns: 15 | ~513, ~527 |
| `crates/swarm-agents/src/agents/manager.rs` | Added `plan_slot` field to ManagerWorkers; wired `PlanParallelWorkTool` into builder | ~3 blocks |
| `crates/swarm-agents/src/agents/mod.rs` | Added `with_plan_slot()` builder method to AgentFactory | ~1 method |
| `crates/swarm-agents/src/tools/plan_parallel_tool.rs` | **NEW** (175 LOC): `PlanSlot` type, `PlanParallelWorkTool`, validation logic | new file |
| `crates/swarm-agents/src/tools/workpad_tool.rs` | **NEW** (313 LOC): Workpad file ops, `AnnounceTool`, `CheckAnnouncementsTool` | new file |
| `crates/swarm-agents/src/tools/bundles.rs` | Added workpad tools to `subtask_worker_tools` bundle | ~2 lines |

### Architecture: Parallel Dispatch Flow

```text
1. Cloud Manager receives issue
   └─> Calls `plan_parallel_work(subtasks: [S1, S2, ...])`
       └─> Validates: no file overlap, min 2 subtasks, integration file rules
           └─> Deposits SubtaskPlan in PlanSlot

2. Orchestrator checks PlanSlot after manager completes
   └─> If plan.len() >= 2 && concurrent_subtasks:
       └─> Init workpad: `.swarm-workpad.jsonl` (empty)
           └─> Spawn JoinSet with Semaphore(max_concurrent=2)
               ├─> Worker-1 executes S1 (target_files partition)
               │   ├─> Reads context + target files
               │   ├─> Makes code changes
               │   └─> Calls `announce()` if interface changed
               │
               ├─> Worker-2 executes S2 (target_files partition)
               │   ├─> Reads context + target files
               │   ├─> Calls `check_announcements()` before final edits
               │   └─> Adapts code if Worker-1 changed public interface
               │
               └─> Both complete
                   └─> Workpad shows announcements for debugging

3. Post-dispatch: Verify, retry on failure, fixer post-pass
   └─> If only 1 worker succeeded: run fixer on integration files only
   └─> If both failed: escalate to Council
   └─> Verifier runs on combined result
```

### Manager Planning Tool

**Tool Name**: `plan_parallel_work`
**Input**: `{"subtasks": [{"id": "s1", "target_files": [...], "objective": "..."}, ...], "summary": "..."}`
**Validation**:
- At least 2 subtasks required
- No file overlap across subtasks
- Integration files (`Cargo.toml`, `mod.rs`, `lib.rs`, `main.rs`) appear in ≤1 subtask

**Behavior**: Deposits validated plan in `PlanSlot` for orchestrator pickup.

### Workpad Format

**File**: `.swarm-workpad.jsonl` (created at dispatch, one JSON per line)

```jsonl
{"worker": "subtask-1", "type": "interface_change", "file": "types.rs", "detail": "Changed FooConfig fields"}
{"worker": "subtask-2", "type": "done", "file": "handler.rs"}
{"worker": "subtask-1", "type": "public_fn_signature_change", "file": "lib.rs", "detail": "Added async fn new_endpoint()"}
```

**Tools**:
- `announce(message: String)` — Appends line to workpad; called by worker when interface changes
- `check_announcements()` → Returns unread entries; workers call before final edits

---

## How to Verify Everything Works

### Quick Checks (2 min)

```bash
# 1. Tests pass
cargo test -p swarm-agents --lib
# Expected: test result: ok. 498 passed; 0 failed

# 2. Build clean
cargo build -p swarm-agents
# Expected: Finished `dev` profile in 0.46s (or similar)

# 3. Clippy clean
cargo clippy --workspace -- -D warnings
# Expected: No errors

# 4. Endpoints healthy
curl -s http://10.0.0.22:8081/health  # Scout/Fast
curl -s http://10.0.0.20:8081/health  # Coder
# Expected: {"status": "ok"}
```

### Dogfood Validation (30-60 min per run)

```bash
# On ai-proxy (brian@100.105.113.58):
cd ~/code/beefcake-swarm

# Single test-probe run (5 min)
SWARM_CLOUD_API_KEY=$SWARM_CLOUD_API_KEY \
  SWARM_CLOUD_URL=http://localhost:8317/v1 \
  SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 \
  timeout 120 bash scripts/run-swarm.sh \
    --issue test-probe --objective 'Reply with OK'

# Multi-issue dogfood loop (2-3 hours per batch)
nohup bash -c 'export SWARM_CLOUD_API_KEY=$SWARM_CLOUD_API_KEY \
    SWARM_CLOUD_URL=http://localhost:8317/v1 \
    SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 \
    RUST_LOG=info && \
  ./scripts/dogfood-loop.sh \
    --issue-list "issue-id-1 issue-id-2 issue-id-3" \
    --cooldown 30 --max-runs 3' \
  > ~/dogfood-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

### What to Look For in Logs

**Success indicators**:
```
INFO swarm_agents: Manager submitted parallel work plan — dispatching concurrent workers
INFO swarm_agents: Dispatching concurrent subtasks (count=2)
INFO swarm_agents: All subtask workers completed (succeeded=2, failed=0)
```

**Failure diagnosis**:
```
# All workers failed → Check for Fix #2 (escalation reset)
INFO swarm_agents: All subtasks failed → escalating to Council

# Single subtask → Check for Fix #4 (routing)
INFO swarm_agents: Single-subtask plan → escalating to Council

# Manager delegation → Check for Fix #3 (git-backed detection)
INFO swarm_agents: Manager delegated work (git changes detected)

# Budget exhaustion → Check for Fix #1 (write deadline)
WARN subtask: Worker budget exhausted: turns_without_write=8, max=8
```

---

## Known Gotchas & Troubleshooting

### Inference Endpoints Down
**Symptom**: Preflight check fails instantly, all runs fail in 5 seconds.
**Fix**: Restart SLURM jobs:
```bash
ssh root@10.0.0.5
sbatch /cluster/shared/scripts/llama-cpp/run-27b-256k.slurm   # Scout/Fast
sbatch /cluster/shared/scripts/llama-cpp/run-122b-rpc.slurm    # Coder
# Wait 30-60 sec for endpoints to load models and become ready
```

### DNS Issues on ai-proxy
**Symptom**: `curl http://vasp-03:8081/health` fails with "Could not resolve host".
**Workaround**: Use IP addresses: `curl http://10.0.0.22:8081/health`
**Root**: Hostname resolution not configured on ai-proxy; use IPs directly.

### Plan Slot Deadlock
**Symptom**: Orchestrator hangs waiting for plan from manager.
**Debug**: Check if manager called `plan_parallel_work` tool. If not, manager may have failed or chosen sequential path.
**Fix**: Ensure manager tools include `PlanParallelWorkTool` (verified in `build_cloud_manager`).

### Single Subtask Loop
**Symptom**: Dogfood succeeds but uses Worker tier (logs show no "Manager submitted parallel work plan").
**Cause**: Manager's local planner (when cloud unavailable) returns 1 subtask; Fix #4 escalates to Council.
**Next Step**: Consider passing full issue context (not just title) to planner to improve decomposition.

---

## Next Steps for the Next Agent

### Immediate (Validate Fixes)
1. **Run dogfood with multi-file issues** that will trigger parallel dispatch
   - Look for at least one run with 2+ subtasks successfully executing in parallel
   - Verify workpad contains inter-worker announcements
   - Confirm verifier passes on combined output

2. **Monitor metrics**:
   - Speedup: Do 2 concurrent workers finish in ~parallel time vs 2x sequential?
   - Iteration count: Do all-fail cases escalate immediately (Fix #2)?
   - Write success: Do workers complete with has_written=true (Fix #3)?

### Short Term (Polish)
3. **Improve planner decomposition** (optional but recommended):
   - Currently planner gets issue title only; pass full issue description
   - This may help with more aggressive multi-subtask decomposition
   - Verify against dogfood results: if most runs return 1 subtask, this is a quick win

4. **Add observability** (if needed based on dogfood):
   - Per-worker timings: how long does each subtask take?
   - File conflict detection: log any integration files assigned to multiple subtasks (should be zero)
   - Workpad entries per worker: count announcements to measure cross-worker awareness

### Medium Term (Phase 4+)
5. **Deferred Issue #7**: Manager oversight via `check_parallel_status` tool
   - Only implement if workers run >5 minutes and manager needs mid-flight visibility
   - Expected: not needed for current 2-minute workers on 2-node cluster

6. **Scale to 4+ workers** (if 2-node cluster expanded):
   - Upgrade from simple JSONL workpad to full Codex-style tool-backed blackboard
   - Add structured queries for inter-worker lookups
   - Add filtering/partitioning of announcements per worker

---

## Critical Reminders

⚠️ **Fix #3 is the Trickiest**: Manager delegation detection relies on trusting `git_has_changes` as source of truth because the RuntimeAdapter never sees nested `edit_file` calls. If git changes are somehow lost (rare), the detection fails. Monitor dogfood closely for any "has_written=false" on manager iterations.

⚠️ **Planner Decomposition**: The system works best when manager returns 2+ subtasks. If planner consistently returns 1 subtask for multi-file issues, debug the planner (Issue #2b: pass full issue context). Don't add workarounds; fix the root cause.

⚠️ **Write Deadline Tuning**: `max_turns_without_write=8` and `max_turns=15` are tuned for 2-file subtasks with context expansion. If workers consistently hit the budget on larger subtasks, consider increasing further. Monitor `WARN subtask: Worker budget exhausted` messages.

---

## Quick Reference

```bash
# Health checks
curl -s http://10.0.0.22:8081/health  # Scout/Fast
curl -s http://10.0.0.20:8081/health  # Coder

# Build & test
cargo build -p swarm-agents && cargo test -p swarm-agents && cargo clippy --workspace -- -D warnings

# Latest commits
git log --oneline -10

# Check concurrent dispatch config
grep "concurrent_subtasks" crates/swarm-agents/src/config.rs

# Run single test-probe
ssh brian@100.105.113.58 "cd ~/code/beefcake-swarm && SWARM_CLOUD_API_KEY=\$SWARM_CLOUD_API_KEY SWARM_CLOUD_URL=http://localhost:8317/v1 SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 timeout 120 bash scripts/run-swarm.sh --issue test-probe --objective 'Reply with OK'"

# Monitor dogfood logs
ssh brian@100.105.113.58 "tail -f ~/dogfood-*.log"

# Kill stale worktrees
rm -rf /tmp/beefcake-wt/* && git worktree prune
```

---

**Handoff prepared**: 2026-03-12, 11:45 UTC
**By**: Claude Haiku
**Status**: Ready for next agent to validate via dogfood
