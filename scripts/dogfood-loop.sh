#!/usr/bin/env bash
# dogfood-loop.sh — Run swarm-agents on beads issues, optionally in parallel.
#
# Usage:
#   ./scripts/dogfood-loop.sh                    # Pick from bd ready (default)
#   ./scripts/dogfood-loop.sh --max-runs 5       # Stop after 5 runs
#   ./scripts/dogfood-loop.sh --cooldown 30      # 30-second cooldown between runs
#   ./scripts/dogfood-loop.sh --parallel 3       # Run up to 3 issues concurrently
#   ./scripts/dogfood-loop.sh --issue-list "beefcake-w70b.3.5.1 beefcake-w70b.6.4.2"
#                                                # Work specific issues in order
#
# Parallel mode launches N issues simultaneously, each in its own worktree.
# Issues are distributed across cluster nodes (vasp-01/02/03) via the
# round-robin of Qwen3.5-397B endpoints configured in run-swarm.sh.
#
# Environment:
#   SWARM_CLOUD_API_KEY    Required (passed through to run-swarm.sh)
#   DOGFOOD_LOG_DIR        Log directory (default: ./logs/dogfood)
#   DOGFOOD_MAX_RUNS       Max iterations (default: unlimited)
#   DOGFOOD_COOLDOWN       Seconds between batches (default: 30)
#   DOGFOOD_ISSUE_LIST     Space-separated issue IDs to work in order
#   DOGFOOD_PARALLEL       Concurrent issue limit (default: 1)
#   DOGFOOD_STRINGER_INTERVAL  Seconds between stringer backlog refreshes (default: 86400)
#   DOGFOOD_WORKER_FIRST   Prefer worker-first routing for dogfood runs (default: 1)
#   DOGFOOD_ALLOW_DIRTY_TARGET  Allow tracked changes in --repo-root target (default: 0)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Emergency stop sentinel: touch /tmp/beefcake-dogfood-stop to prevent all new starts.
# This is checked before acquiring the lockfile so even parallel children abort immediately.
if [[ -f "/tmp/beefcake-dogfood-stop" ]]; then
    echo "[dogfood-loop] Stop sentinel /tmp/beefcake-dogfood-stop exists; aborting." >&2
    exit 0
fi

# Source persistent env file if it exists (ensures API keys are available
# in nohup/cron/systemd contexts where ~/.bashrc is not sourced).
if [[ -f "$HOME/.swarm-env" ]]; then
    set -a; source "$HOME/.swarm-env"; set +a
fi

# --- Configuration ---
MAX_RUNS="${DOGFOOD_MAX_RUNS:-0}"       # 0 = unlimited
COOLDOWN="${DOGFOOD_COOLDOWN:-30}"
LOG_DIR="${DOGFOOD_LOG_DIR:-${REPO_ROOT}/logs/dogfood}"
ISSUE_LIST="${DOGFOOD_ISSUE_LIST:-}"
PARALLEL="${DOGFOOD_PARALLEL:-1}"
ISSUE_QUERY_BIN="${DOGFOOD_BEADS_BIN:-${SCRIPT_DIR}/bd-safe.sh}"
DISCOVER=0                              # --discover: auto-fetch new issues when list exhausted
TARGET_REPO=""                          # --repo-root: target an external repo (default: self)
MAX_ISSUE_FAILURES="${DOGFOOD_MAX_ISSUE_FAILURES:-3}"  # defer after N consecutive failures
STRINGER_INTERVAL="${DOGFOOD_STRINGER_INTERVAL:-86400}"
DOGFOOD_WORKER_FIRST="${DOGFOOD_WORKER_FIRST:-1}"
ALLOW_DIRTY_TARGET="${DOGFOOD_ALLOW_DIRTY_TARGET:-0}"
ISSUE_SELECTION="${SWARM_ISSUE_SELECTION:-ucb1}"  # ucb1, priority, random

# CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runs)    MAX_RUNS="$2"; shift 2 ;;
    --cooldown)    COOLDOWN="$2"; shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --issue-list)  ISSUE_LIST="$2"; shift 2 ;;
    --parallel)    PARALLEL="$2"; shift 2 ;;
    --discover)    DISCOVER=1; shift ;;
    --repo-root)   TARGET_REPO="$2"; shift 2 ;;
    --max-issue-failures) MAX_ISSUE_FAILURES="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# When targeting an external repo, bd commands run in that repo's directory
# and run-swarm.sh gets --repo-root passed through.
EXTRA_SWARM_ARGS=()
BD_RUN_DIR="$REPO_ROOT"
if [[ -n "$TARGET_REPO" ]]; then
  TARGET_REPO="$(cd "$TARGET_REPO" && pwd)"  # absolutize
  BD_RUN_DIR="$TARGET_REPO"
  EXTRA_SWARM_ARGS+=("--repo-root" "$TARGET_REPO")

  if [[ "$ALLOW_DIRTY_TARGET" != "1" ]]; then
    mapfile -t _tracked_target_changes < <(
      git -C "$TARGET_REPO" status --porcelain --untracked-files=no 2>/dev/null |
        sed -E 's/^...//'
    )
    if [[ ${#_tracked_target_changes[@]} -gt 0 ]]; then
      echo "ERROR: Target repo has tracked local changes and is unsafe for landing merges:" >&2
      printf '  %s\n' "${_tracked_target_changes[@]}" >&2
      echo "Use a clean integration worktree as --repo-root, or set DOGFOOD_ALLOW_DIRTY_TARGET=1 to override." >&2
      exit 1
    fi
  fi
fi

mkdir -p "$LOG_DIR"

# ── Lockfile: prevent overlapping loop instances ──
# Uses flock(1) for atomic, race-free locking. The lock is automatically
# released when the process exits (including crashes/signals).
# Lockfile includes target repo name to allow parallel loops on different repos.
#
# IMPORTANT: Never delete the lockfile! flock works on inodes, not paths.
# If you rm the lockfile and recreate it, the new file gets a new inode
# and flock sees it as a different lock — allowing zombie accumulation.
# To stop the loop: kill the PID written inside the lockfile, or use
# `flock -u 200` to release the lock explicitly.
_LOCK_SUFFIX=""
if [[ -n "$TARGET_REPO" ]]; then
  _LOCK_SUFFIX="-$(basename "$TARGET_REPO")"
fi
LOCKFILE="/tmp/dogfood-loop${_LOCK_SUFFIX}.lock"
STRINGER_STAMP_FILE="/tmp/dogfood-loop${_LOCK_SUFFIX}.stringer.last_run"
# Read existing PID BEFORE opening fd 200 (which truncates the file).
_EXISTING_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
# Open fd 200 for flock. O_WRONLY|O_CREAT but NOT O_TRUNC — use >> to append.
# This preserves existing content so other instances can still read the PID.
exec 200>>"$LOCKFILE"
if ! flock -n 200; then
    # Stale lock check: if the PID in the lockfile is dead, the lock is stale.
    if [[ "$_EXISTING_PID" =~ ^[0-9]+$ ]] && ! kill -0 "$_EXISTING_PID" 2>/dev/null; then
        echo "WARNING: Stale lock (pid=$_EXISTING_PID is dead). Reclaiming lock." >&2
        exec 200>&-
        exec 200>>"$LOCKFILE"
        if ! flock -n 200; then
            echo "ERROR: Could not reclaim stale lock. Another instance may have started." >&2
            exit 1
        fi
    else
        echo "ERROR: Another dogfood-loop is already running (pid=$_EXISTING_PID)." >&2
        echo "Kill it first: kill $_EXISTING_PID" >&2
        exit 1
    fi
fi
# Write our PID to the lockfile (truncate first to remove stale content).
echo $$ > "$LOCKFILE"

# Process group cleanup is set up later (after CHILD_PIDS is defined).
# See the `cleanup` function and `trap cleanup ...` below.

# Note: sccache/CARGO_TARGET_DIR are set by run-swarm.sh (gated on target language)

# --- Telemetry ---
DOGFOOD_START=$(date +%s)
RUN_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SUMMARY_FILE="${LOG_DIR}/dogfood-summary.jsonl"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Run bd commands in the target repo directory (for multi-repo support).
bd_cmd() {
  (cd "$BD_RUN_DIR" && "$ISSUE_QUERY_BIN" "$@")
}

record_run() {
  local run_num="$1" issue_id="$2" exit_code="$3" elapsed="$4" log_file="$5"
  printf '{"run":%d,"issue":"%s","exit_code":%d,"elapsed_s":%d,"timestamp":"%s","log":"%s"}\n' \
    "$run_num" "$issue_id" "$exit_code" "$elapsed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$log_file" \
    >> "$SUMMARY_FILE"
}

maybe_run_stringer() {
  [[ "$DISCOVER" -eq 1 ]] || return 0

  local stringer_script="$SCRIPT_DIR/stringer-to-beads.sh"
  [[ -x "$stringer_script" ]] || return 0

  local now last_run elapsed remaining
  now=$(date +%s)
  last_run=0
  if [[ -f "$STRINGER_STAMP_FILE" ]]; then
    last_run=$(cat "$STRINGER_STAMP_FILE" 2>/dev/null || echo "0")
  fi

  if [[ "$last_run" =~ ^[0-9]+$ ]]; then
    elapsed=$((now - last_run))
    if (( elapsed < STRINGER_INTERVAL )); then
      remaining=$((STRINGER_INTERVAL - elapsed))
      log "  [discover] Skipping stringer-to-beads.sh (${remaining}s until next refresh)"
      return 0
    fi
  fi

  log "  [discover] Running stringer-to-beads.sh to seed backlog..."
  if bash "$stringer_script" --max-new 30 2>&1 | grep -E '^\[stringer|Created|WARN' | sed 's/^/    /'; then
    printf '%s\n' "$now" > "$STRINGER_STAMP_FILE"
  fi
}

# --- Reconciliation: fix zombie issues between batches ---
# Runs periodically to close issues with merged PRs and reset stale in_progress.
RECONCILE_STAMP_FILE="/tmp/dogfood-loop${_LOCK_SUFFIX}.reconcile.last_run"
RECONCILE_INTERVAL="${DOGFOOD_RECONCILE_INTERVAL:-600}"  # every 10 minutes

ENRICH_STAMP_FILE="/tmp/dogfood-loop${_LOCK_SUFFIX}.enrich.last_run"
ENRICH_INTERVAL="${DOGFOOD_ENRICH_INTERVAL:-604800}"  # weekly (7 days)

maybe_reconcile() {
  local reconcile_script="$SCRIPT_DIR/reconcile-issues.sh"
  [[ -x "$reconcile_script" ]] || return 0

  local now last_run elapsed
  now=$(date +%s)
  last_run=0
  if [[ -f "$RECONCILE_STAMP_FILE" ]]; then
    last_run=$(cat "$RECONCILE_STAMP_FILE" 2>/dev/null || echo "0")
  fi

  if [[ "$last_run" =~ ^[0-9]+$ ]]; then
    elapsed=$((now - last_run))
    if (( elapsed < RECONCILE_INTERVAL )); then
      return 0
    fi
  fi

  log "  [reconcile] Running reconcile-issues.sh..."
  local extra_args=()
  if [[ -n "$TARGET_REPO" ]]; then
    extra_args+=(--repo-root "$TARGET_REPO")
  fi
  if bash "$reconcile_script" "${extra_args[@]}" 2>&1 | sed 's/^/    /'; then
    printf '%s\n' "$now" > "$RECONCILE_STAMP_FILE"
  fi
}

# --- Cognition enrichment: query NotebookLM for insights weekly ---
maybe_enrich_cognition() {
  local enrich_script="$SCRIPT_DIR/enrich-cognition.sh"
  [[ -x "$enrich_script" ]] || return 0

  local now last_run elapsed
  now=$(date +%s)
  last_run=0
  if [[ -f "$ENRICH_STAMP_FILE" ]]; then
    last_run=$(cat "$ENRICH_STAMP_FILE" 2>/dev/null || echo "0")
  fi

  if [[ "$last_run" =~ ^[0-9]+$ ]]; then
    elapsed=$((now - last_run))
    if (( elapsed < ENRICH_INTERVAL )); then
      return 0
    fi
  fi

  log "  [enrich] Running enrich-cognition.sh (weekly cadence)..."
  local extra_args=()
  if [[ -n "$TARGET_REPO" ]]; then
    extra_args+=(--repo-root "$TARGET_REPO")
  fi
  if bash "$enrich_script" "${extra_args[@]}" 2>&1 | sed 's/^/    /'; then
    printf '%s\n' "$now" > "$ENRICH_STAMP_FILE"
  fi
}

# --- Circuit breaker: defer issues that fail too many times ---
# Counts consecutive recent failures for an issue in the summary JSONL.
# Returns 0 (true) if the issue has exceeded MAX_ISSUE_FAILURES.
issue_is_exhausted() {
  local issue_id="$1"
  [[ ! -f "$SUMMARY_FILE" ]] && return 1

  local consecutive_failures
  consecutive_failures=$(python3 -c "
import json, sys
fails = 0
# Read all runs for this issue, count consecutive failures from the end
runs = []
with open('$SUMMARY_FILE') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('issue') == '$issue_id':
                runs.append(r)
        except: pass
# Count backwards from most recent
for r in reversed(runs):
    if r.get('exit_code', 1) != 0:
        fails += 1
    else:
        break
print(fails)
" 2>/dev/null || echo "0")

  [[ "$consecutive_failures" -ge "$MAX_ISSUE_FAILURES" ]]
}

defer_exhausted_issue() {
  local issue_id="$1" consecutive_failures="$2"
  log "  Circuit breaker: $issue_id failed $consecutive_failures consecutive times (limit: $MAX_ISSUE_FAILURES) — deferring"
  bd_cmd update "$issue_id" --status=deferred 2>/dev/null || true
}

# --- Pre-flight ---
log "Dogfood loop starting"
log "  Engine:   $REPO_ROOT"
if [[ -n "$TARGET_REPO" ]]; then
  log "  Target:   $TARGET_REPO (external repo)"
else
  log "  Target:   $REPO_ROOT (self-dogfood)"
fi
log "  Logs:     $LOG_DIR"
log "  Cooldown: ${COOLDOWN}s"
log "  Parallel: $PARALLEL"
log "  Max runs: $([ "$MAX_RUNS" -eq 0 ] && echo 'unlimited' || echo "$MAX_RUNS")"
log "  Discover: $([ "$DISCOVER" -eq 1 ] && echo 'ON (auto-fetch new issues)' || echo 'OFF')"
log "  Stringer refresh: ${STRINGER_INTERVAL}s"
log "  Worker-first: ${DOGFOOD_WORKER_FIRST}"
log "  Selection:  ${ISSUE_SELECTION}"

if [[ -n "$ISSUE_LIST" ]]; then
  read -ra ISSUES <<< "$ISSUE_LIST"
  log "  Issues:   ${ISSUES[*]} (${#ISSUES[@]} total)"
else
  ISSUES=()
  log "  Issues:   auto (from bdh ready)"
fi

# Verify toolchain
if ! command -v cargo &>/dev/null; then
  echo "ERROR: cargo not found. Install Rust toolchain first." >&2
  exit 1
fi
if ! command -v "$ISSUE_QUERY_BIN" &>/dev/null; then
  echo "WARNING: $ISSUE_QUERY_BIN not found. Issue objectives will fall back to issue IDs." >&2
fi

parse_bdh_json() {
  python3 -c '
import json, sys

doc = json.load(sys.stdin)
payload = doc.get("bd_stdout", doc) if isinstance(doc, dict) else doc

if sys.argv[1] == "field":
    # bdh show --json returns a dict; bdh list/ready --json returns a list
    if isinstance(payload, dict):
        item = payload
    elif isinstance(payload, list) and payload:
        item = payload[0]
    else:
        item = {}
    value = item.get(sys.argv[2], "") if isinstance(item, dict) else ""
    limit = int(sys.argv[3])
    if isinstance(value, str):
        print(value[:limit])
    else:
        print(value)
elif sys.argv[1] == "count":
    # Handle both list and dict-with-tasks-key formats
    if isinstance(payload, list):
        print(len(payload))
    elif isinstance(payload, dict) and "tasks" in payload:
        print(len(payload["tasks"]))
    else:
        print(0)
elif sys.argv[1] == "ids":
    # bdh ready --json returns {"tasks": [...]} or a bare list
    # Each task has "task_ref" (e.g. "beefcake-swarm-042") not "id"
    items = payload
    if isinstance(payload, dict) and "tasks" in payload:
        items = payload["tasks"]
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                ref = item.get("task_ref") or item.get("id") or ""
                if ref:
                    print(ref)
' "$@"
}

# --- UCB1-based issue selection (ASI-Evolve pattern) ---
# Scores ready issues balancing exploitation (category success rate)
# vs exploration (under-attempted categories).
select_issues_ucb1() {
  local top_n="${1:-3}"
  local ucb1_script="$SCRIPT_DIR/ucb1-select.py"
  local exp_db="${BD_RUN_DIR}/.swarm/experiment_history.jsonl"
  local summary_db="$SUMMARY_FILE"

  if [[ -f "$ucb1_script" ]] && command -v python3 &>/dev/null; then
    local selected
    selected=$(python3 "$ucb1_script" \
      --experiment-db "$exp_db" \
      --summary-db "$summary_db" \
      --top-n "$top_n" 2>/dev/null)
    if [[ -n "$selected" ]]; then
      echo "$selected"
      return 0
    fi
  fi

  # Fallback: use bd ready (priority order)
  bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null | head -n "$top_n" | tr '\n' ' '
}

# Select issues based on SWARM_ISSUE_SELECTION strategy.
# Dispatches to ucb1, priority (bd ready), or random selection.
select_issues() {
  local top_n="${1:-3}"
  case "$ISSUE_SELECTION" in
    ucb1)
      select_issues_ucb1 "$top_n"
      ;;
    random)
      # Shuffle bd ready output
      bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null | sort -R | head -n "$top_n" | tr '\n' ' '
      ;;
    priority|*)
      # Default: bd ready priority order
      bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null | head -n "$top_n" | tr '\n' ' '
      ;;
  esac
}

# --- Run a single issue (called from parallel dispatch) ---
run_issue() {
  local run_num="$1" issue_id="$2"
  local run_log="${LOG_DIR}/run-${run_num}-${issue_id}-$(date +%Y%m%d-%H%M%S).log"
  local issue_title issue_args=()
  local run_start run_end elapsed exit_code

  # Early check: skip if issue already closed (prevents re-processing resolved issues)
  local issue_status
  issue_status=$(bd_cmd show "$issue_id" --json 2>/dev/null | parse_bdh_json field status 50 2>/dev/null || echo "")
  if [[ "$issue_status" == "closed" ]]; then
    log "  [run $run_num] SKIP issue=$issue_id (already closed)"
    return 0
  fi

  # We rely on swarm-agents to fetch the full issue packet directly via BeadsBridge.
  issue_args=(--issue "$issue_id" "${EXTRA_SWARM_ARGS[@]}")

  # Fetch title just for the warning check
  issue_title=$(bd_cmd show "$issue_id" --json 2>/dev/null | parse_bdh_json field title 1000 2>/dev/null || echo "")

  # Quality gate: issues with very short titles tend to fail (r=0.545 correlation)
  if [[ ${#issue_title} -lt 100 ]]; then
    log "  [run $run_num] WARN: short title (${#issue_title} chars) — may have low success rate"
  fi

  log "  [run $run_num] Starting issue=$issue_id log=$run_log"

  run_start=$(date +%s)
  set +e
  (
    cd "$REPO_ROOT"
    SWARM_WORKER_FIRST_ENABLED="${SWARM_WORKER_FIRST_ENABLED:-$DOGFOOD_WORKER_FIRST}" \
      bash scripts/run-swarm.sh "${issue_args[@]}" 2>&1
  ) > "$run_log" 2>&1
  exit_code=$?
  set -e
  run_end=$(date +%s)
  elapsed=$((run_end - run_start))

  if [[ $exit_code -eq 0 ]]; then
    log "  [run $run_num] SUCCESS issue=$issue_id (${elapsed}s)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

    # Post-merge lint janitor: auto-fix trivial lint/format/type issues
    # WITHOUT creating beads issues. Only truly non-auto-fixable findings
    # get promoted to issues (with dedup). Replaces generate-issues.sh
    # which was creating duplicate issues and wasting swarm cycles.
    if [[ -n "$TARGET_REPO" && -x "$REPO_ROOT/scripts/lint-janitor.sh" ]]; then
      log "  [run $run_num] Post-merge: running lint janitor..."
      MAX_ISSUES=3 bash "$REPO_ROOT/scripts/lint-janitor.sh" "$BD_RUN_DIR" >> "$run_log" 2>&1 || true
    elif [[ -n "$TARGET_REPO" && -x "$REPO_ROOT/scripts/generate-issues.sh" ]]; then
      # Fallback to generate-issues.sh if lint-janitor not available
      log "  [run $run_num] Post-merge: scanning for new issues (legacy)..."
      MAX_ISSUES=5 bash "$REPO_ROOT/scripts/generate-issues.sh" "$BD_RUN_DIR" >> "$run_log" 2>&1 || true
    fi

    # Post-merge benchmark gate: verify physics correctness on GPU.
    # Classifies the merge (physics-touching vs cosmetic), then runs
    # the appropriate benchmark tier on vasp-03.
    if [[ -n "$TARGET_REPO" && -x "$REPO_ROOT/scripts/post-merge-benchmark.sh" ]]; then
      change_type=$(bash "$REPO_ROOT/scripts/post-merge-benchmark.sh" classify "$BD_RUN_DIR" 2>/dev/null || echo "skip")
      if [[ "$change_type" != "skip" ]]; then
        log "  [run $run_num] Running $change_type benchmark gate..."
        if bash "$REPO_ROOT/scripts/post-merge-benchmark.sh" run "$BD_RUN_DIR" "$change_type" "$issue_id" >> "$run_log" 2>&1; then
          log "  [run $run_num] Benchmark PASSED ($change_type)"
        else
          log "  [run $run_num] Benchmark FAILED ($change_type) — creating regression issue"
          bd_cmd create \
            --title="BUG: Benchmark regression after $issue_id merge" \
            --description="Post-merge $change_type benchmark failed after merging $issue_id (sha=$(git -C "$BD_RUN_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)). The change may have broken physics correctness. Check benchmark output on vasp-03 at /scratch/cf-libs-bench/repo/output/benchmark_gate/ for results." \
            --type=bug --priority=1 2>/dev/null || true
          FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
      else
        log "  [run $run_num] Benchmark skipped (cosmetic-only changes)"
      fi
    fi
  else
    log "  [run $run_num] FAILED  issue=$issue_id exit=$exit_code (${elapsed}s)"
    tail -3 "$run_log" | while IFS= read -r line; do
      log "    $line"
    done

    # Cloud council postmortem: diagnose failure, update issue, reopen for retry.
    if [[ -x "$REPO_ROOT/scripts/postmortem-review.sh" ]]; then
      log "  [run $run_num] Running cloud council postmortem..."
      if bash "$REPO_ROOT/scripts/postmortem-review.sh" "$issue_id" "$run_log" >> "$run_log" 2>&1; then
        log "  [run $run_num] Postmortem complete — issue reopened with guidance"
      else
        log "  [run $run_num] Postmortem failed (non-fatal)"
      fi
    fi
  fi

  record_run "$run_num" "$issue_id" "$exit_code" "$elapsed" "$run_log"
  return $exit_code
}

# --- Main loop ---
# Track child PIDs for the trap handler
CHILD_PIDS=()

cleanup() {
  log "Interrupted. Killing ${#CHILD_PIDS[@]} tracked children + process group..."
  # Kill tracked children first (specific PIDs from parallel dispatch)
  for pid in "${CHILD_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Then kill entire process group to catch any untracked descendants
  # (run-swarm.sh, swarm-agents, cargo, etc.) that would otherwise zombie.
  kill -- -$$ 2>/dev/null || true
  wait 2>/dev/null || true
  log "Runs=$RUN_COUNT Success=$SUCCESS_COUNT Fail=$FAIL_COUNT"
  exit 130
}
trap cleanup INT TERM EXIT

if [[ "$PARALLEL" -le 1 ]]; then
  # --- Serial mode (original behavior) ---
  while true; do
    RUN_COUNT=$((RUN_COUNT + 1))

    if [[ "$MAX_RUNS" -gt 0 && "$RUN_COUNT" -gt "$MAX_RUNS" ]]; then
      log "Reached max runs ($MAX_RUNS). Stopping."
      break
    fi

    ISSUE_ID="auto"
    if [[ ${#ISSUES[@]} -gt 0 ]]; then
      IDX=$((RUN_COUNT - 1))
      if [[ $IDX -ge ${#ISSUES[@]} ]]; then
        if [[ "$DISCOVER" -eq 1 ]]; then
          # Discover new issues (UCB1-ranked if enabled, else bd ready)
          log "Issue list exhausted — discovering new issues (strategy=$ISSUE_SELECTION)..."
          mapfile -t NEW_ISSUES < <(select_issues 20 | tr ' ' '\n' || true)
          if [[ ${#NEW_ISSUES[@]} -gt 0 ]]; then
            # Dedup: only add issues not already in the ISSUES array
            for _new_id in "${NEW_ISSUES[@]}"; do
              _already_in=0
              for _existing_id in "${ISSUES[@]}"; do
                if [[ "$_new_id" == "$_existing_id" ]]; then _already_in=1; break; fi
              done
              if [[ $_already_in -eq 0 ]]; then ISSUES+=("$_new_id"); fi
            done
            log "  Discovered ${#NEW_ISSUES[@]} new issues (after dedup: ${#ISSUES[@]} total): ${NEW_ISSUES[*]}"
            ISSUE_ID="${ISSUES[$IDX]}"
          else
            log "  No new ready issues. Waiting ${COOLDOWN}s before retry..."
            sleep "$COOLDOWN"
            continue
          fi
        else
          log "All ${#ISSUES[@]} issues processed. Stopping."
          break
        fi
      else
        ISSUE_ID="${ISSUES[$IDX]}"
      fi
    else
      # Pick from ready issues using Rust-native reformulation-aware selection.
      # Falls back to bash circuit breaker if `swarm-agents pick-next` is unavailable.
      ISSUE_ID=""
      PICK_NEXT_JSON=$(cd "$BD_RUN_DIR" && cargo run --manifest-path "$REPO_ROOT/Cargo.toml" -p swarm-agents --quiet -- pick-next --json 2>/dev/null || true)
      if [[ -n "$PICK_NEXT_JSON" ]]; then
        ISSUE_ID=$(echo "$PICK_NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)
        if [[ -n "$ISSUE_ID" ]]; then
          log "  pick-next selected: $ISSUE_ID"
        fi
      fi

      # Fallback: UCB1/priority selection with circuit breaker
      if [[ -z "$ISSUE_ID" ]]; then
        while IFS= read -r candidate; do
          [[ -z "$candidate" ]] && continue
          if issue_is_exhausted "$candidate"; then
            defer_exhausted_issue "$candidate" "$MAX_ISSUE_FAILURES"
            continue
          fi
          ISSUE_ID="$candidate"
          break
        done < <(select_issues 10 | tr ' ' '\n' || true)
      fi

      if [[ -z "$ISSUE_ID" ]]; then
        if [[ "$DISCOVER" -eq 1 ]]; then
          log "No ready issues (all exhausted or none available). Waiting ${COOLDOWN}s before retry..."
          sleep "$COOLDOWN"
          continue
        fi
        log "No more ready issues. Stopping."
        break
      fi
    fi

    # Final circuit breaker check for issues from the explicit list too
    if [[ "$ISSUE_ID" != "auto" ]] && issue_is_exhausted "$ISSUE_ID"; then
      defer_exhausted_issue "$ISSUE_ID" "$MAX_ISSUE_FAILURES"
      log "  Skipping exhausted issue $ISSUE_ID"
      sleep 5
      continue
    fi

    log "=== Run $RUN_COUNT: issue=$ISSUE_ID ==="
    if run_issue "$RUN_COUNT" "$ISSUE_ID"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Check if more work exists (skip in discover mode — it loops forever)
    if [[ ${#ISSUES[@]} -eq 0 && "$DISCOVER" -eq 0 ]]; then
      READY_COUNT=$(bd_cmd ready --json 2>/dev/null | parse_bdh_json count 2>/dev/null || echo "0")
      if [[ "$READY_COUNT" -eq 0 ]]; then
        log "No more ready issues. Stopping."
        break
      fi
      log "  $READY_COUNT issues remaining in bdh ready"
    fi

    # Cooldown (unless done)
    if [[ "$MAX_RUNS" -gt 0 && "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then break; fi
    if [[ ${#ISSUES[@]} -gt 0 && $RUN_COUNT -ge ${#ISSUES[@]} && "$DISCOVER" -eq 0 ]]; then break; fi

    maybe_reconcile
    maybe_enrich_cognition
    log "  Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  done
else
  # --- Parallel mode ---
  log "=== Parallel mode: up to $PARALLEL concurrent issues ==="

  # Build the issue queue (UCB1-ranked if enabled)
  if [[ ${#ISSUES[@]} -eq 0 ]]; then
    # Auto mode: fetch ready issues ranked by selection strategy
    mapfile -t ISSUES < <(select_issues 20 | tr ' ' '\n' || true)
    if [[ ${#ISSUES[@]} -eq 0 ]]; then
      log "No ready issues found. Stopping."
      exit 0
    fi
    log "  Found ${#ISSUES[@]} ready issues (strategy=$ISSUE_SELECTION): ${ISSUES[*]}"
  fi

  # Apply max-runs limit to the queue
  if [[ "$MAX_RUNS" -gt 0 && "$MAX_RUNS" -lt "${#ISSUES[@]}" ]]; then
    ISSUES=("${ISSUES[@]:0:$MAX_RUNS}")
    log "  Trimmed to $MAX_RUNS issues per --max-runs"
  fi

  # Launch issues in batches of $PARALLEL
  IDX=0
  while [[ $IDX -lt ${#ISSUES[@]} || "$DISCOVER" -eq 1 ]]; do
    # When discover is on and we've exhausted the list, fetch new issues
    if [[ $IDX -ge ${#ISSUES[@]} && "$DISCOVER" -eq 1 ]]; then
      log "Issue list exhausted — discovering new issues (strategy=$ISSUE_SELECTION)..."
      mapfile -t NEW_ISSUES < <(select_issues 20 | tr ' ' '\n' || true)
      if [[ ${#NEW_ISSUES[@]} -gt 0 ]]; then
        # Dedup: only add issues not already in the ISSUES array
        for _new_id in "${NEW_ISSUES[@]}"; do
          _already_in=0
          for _existing_id in "${ISSUES[@]}"; do
            if [[ "$_new_id" == "$_existing_id" ]]; then _already_in=1; break; fi
          done
          if [[ $_already_in -eq 0 ]]; then ISSUES+=("$_new_id"); fi
        done
        log "  Discovered ${#NEW_ISSUES[@]} new issues (after dedup: ${#ISSUES[@]} total): ${NEW_ISSUES[*]}"
      else
        log "  No new ready issues. Waiting ${COOLDOWN}s..."
        sleep "$COOLDOWN"
        continue
      fi
    fi
    BATCH_SIZE=$PARALLEL
    REMAINING=$(( ${#ISSUES[@]} - IDX ))
    if [[ $BATCH_SIZE -gt $REMAINING ]]; then
      BATCH_SIZE=$REMAINING
    fi

    # --- Streaming pool: fill slots as they complete (no batch-wait-all) ---
    # Launch initial batch to fill the pool
    log "=== Pool starting: ${BATCH_SIZE} slots, ${#ISSUES[@]} issues queued ==="

    declare -A POOL_PIDS  # pid → issue_id

    # Fill pool to capacity
    for (( i=0; i<BATCH_SIZE && IDX < ${#ISSUES[@]}; i++ )); do
      ISSUE_ID="${ISSUES[$IDX]}"
      RUN_COUNT=$((RUN_COUNT + 1))
      run_issue "$RUN_COUNT" "$ISSUE_ID" &
      POOL_PIDS[$!]="$ISSUE_ID"
      log "  [slot $((i+1))] Started issue=$ISSUE_ID"
      IDX=$((IDX + 1))
    done

    # Process completions and backfill slots
    while [[ ${#POOL_PIDS[@]} -gt 0 ]]; do
      # Wait for any child to exit (bash 4.3+)
      if wait -n 2>/dev/null; then
        # A child exited successfully — find which one
        for pid in "${!POOL_PIDS[@]}"; do
          if ! kill -0 "$pid" 2>/dev/null; then
            # This pid exited — check if success or failure via wait
            if wait "$pid" 2>/dev/null; then
              SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
              log "  [pool] SUCCESS issue=${POOL_PIDS[$pid]}"
            else
              FAIL_COUNT=$((FAIL_COUNT + 1))
              log "  [pool] FAILED  issue=${POOL_PIDS[$pid]}"
            fi
            unset "POOL_PIDS[$pid]"

            # Backfill: launch next issue if queue has more
            if [[ $IDX -lt ${#ISSUES[@]} ]]; then
              ISSUE_ID="${ISSUES[$IDX]}"
              RUN_COUNT=$((RUN_COUNT + 1))
              run_issue "$RUN_COUNT" "$ISSUE_ID" &
              POOL_PIDS[$!]="$ISSUE_ID"
              log "  [pool] Backfill started issue=$ISSUE_ID (${#POOL_PIDS[@]} active)"
              IDX=$((IDX + 1))
            elif [[ "$DISCOVER" -eq 1 && ${#POOL_PIDS[@]} -lt $PARALLEL ]]; then
              # Discover new issues to keep pool full (UCB1-ranked)
              mapfile -t NEW_ISSUES < <(select_issues 10 | tr ' ' '\n' || true)
              if [[ ${#NEW_ISSUES[@]} -gt 0 ]]; then
                # Dedup: only add issues not already in the ISSUES array
                for _new_id in "${NEW_ISSUES[@]}"; do
                  _already_in=0
                  for _existing_id in "${ISSUES[@]}"; do
                    if [[ "$_new_id" == "$_existing_id" ]]; then _already_in=1; break; fi
                  done
                  if [[ $_already_in -eq 0 ]]; then ISSUES+=("$_new_id"); fi
                done
                log "  [pool] Discovered ${#NEW_ISSUES[@]} new issues (after dedup: ${#ISSUES[@]} total)"
                ISSUE_ID="${ISSUES[$IDX]}"
                RUN_COUNT=$((RUN_COUNT + 1))
                run_issue "$RUN_COUNT" "$ISSUE_ID" &
                POOL_PIDS[$!]="$ISSUE_ID"
                log "  [pool] Backfill started issue=$ISSUE_ID (${#POOL_PIDS[@]} active)"
                IDX=$((IDX + 1))
              fi
            fi
            break
          fi
        done
      else
        # wait -n returned non-zero — a child failed
        for pid in "${!POOL_PIDS[@]}"; do
          if ! kill -0 "$pid" 2>/dev/null; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
            log "  [pool] FAILED  issue=${POOL_PIDS[$pid]}"
            unset "POOL_PIDS[$pid]"

            # Backfill on failure too
            if [[ $IDX -lt ${#ISSUES[@]} ]]; then
              ISSUE_ID="${ISSUES[$IDX]}"
              RUN_COUNT=$((RUN_COUNT + 1))
              run_issue "$RUN_COUNT" "$ISSUE_ID" &
              POOL_PIDS[$!]="$ISSUE_ID"
              log "  [pool] Backfill started issue=$ISSUE_ID (${#POOL_PIDS[@]} active)"
              IDX=$((IDX + 1))
            fi
            break
          fi
        done
      fi
    done

    # Pool drained — refresh backlog and reconcile zombie issues, then cooldown.
    maybe_run_stringer || true
    maybe_reconcile
    maybe_enrich_cognition
    log "  Pool drained. Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"

  done
fi

# --- Summary ---
TOTAL_ELAPSED=$(( $(date +%s) - DOGFOOD_START ))
log "=========================================="
log "Dogfood complete"
log "  Total runs:    $RUN_COUNT"
log "  Successes:     $SUCCESS_COUNT"
log "  Failures:      $FAIL_COUNT"
log "  Total time:    ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m)"
log "  Summary file:  $SUMMARY_FILE"
log "=========================================="
