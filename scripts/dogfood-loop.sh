#!/usr/bin/env bash
# dogfood-loop.sh — Run swarm-agents on beads issues, optionally in parallel.
#
# Usage:
#   ./scripts/dogfood-loop.sh                    # Pick from bd ready (default)
#   ./scripts/dogfood-loop.sh --max-runs 5       # Stop after 5 runs
#   ./scripts/dogfood-loop.sh --cooldown 120     # 2-minute cooldown between runs
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
#   DOGFOOD_COOLDOWN       Seconds between batches (default: 60)
#   DOGFOOD_ISSUE_LIST     Space-separated issue IDs to work in order
#   DOGFOOD_PARALLEL       Concurrent issue limit (default: 1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
MAX_RUNS="${DOGFOOD_MAX_RUNS:-0}"       # 0 = unlimited
COOLDOWN="${DOGFOOD_COOLDOWN:-60}"
LOG_DIR="${DOGFOOD_LOG_DIR:-${REPO_ROOT}/logs/dogfood}"
ISSUE_LIST="${DOGFOOD_ISSUE_LIST:-}"
PARALLEL="${DOGFOOD_PARALLEL:-1}"
ISSUE_QUERY_BIN="${DOGFOOD_BEADS_BIN:-bd}"
DISCOVER=0                              # --discover: auto-fetch new issues when list exhausted
TARGET_REPO=""                          # --repo-root: target an external repo (default: self)
MAX_ISSUE_FAILURES="${DOGFOOD_MAX_ISSUE_FAILURES:-3}"  # defer after N consecutive failures

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
fi

mkdir -p "$LOG_DIR"

# ── Lockfile: prevent overlapping loop instances ──
# Uses flock(1) for atomic, race-free locking. The lock is automatically
# released when the process exits (including crashes/signals).
# Lockfile includes target repo name to allow parallel loops on different repos.
_LOCK_SUFFIX=""
if [[ -n "$TARGET_REPO" ]]; then
  _LOCK_SUFFIX="-$(basename "$TARGET_REPO")"
fi
LOCKFILE="/tmp/dogfood-loop${_LOCK_SUFFIX}.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    EXISTING_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
    echo "ERROR: Another dogfood-loop is already running (pid=$EXISTING_PID)." >&2
    echo "Kill it first: kill $EXISTING_PID" >&2
    exit 1
fi
echo $$ > "$LOCKFILE"

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

# --- Run a single issue (called from parallel dispatch) ---
run_issue() {
  local run_num="$1" issue_id="$2"
  local run_log="${LOG_DIR}/run-${run_num}-${issue_id}-$(date +%Y%m%d-%H%M%S).log"
  local issue_title issue_args=()
  local run_start run_end elapsed exit_code

  # Fetch title + description from beads for the objective.
  # Including the description gives find_target_files_by_grep more identifiers
  # to search for (e.g., "edit_file", "verifier") beyond just the title.
  # Try bdh first, fall back to JSONL backup if bdh/bd fails (Dolt server issues).
  issue_title=$(bd_cmd show "$issue_id" --json 2>/dev/null | parse_bdh_json field title 1000 2>/dev/null)
  if [[ -z "$issue_title" || "$issue_title" == "Issue $issue_id" ]]; then
    # Fall back to JSONL backup (always available, no Dolt needed)
    local jsonl_path="${REPO_ROOT}/.beads/backup/issues.jsonl"
    if [[ -f "$jsonl_path" ]]; then
      issue_title=$(python3 -c "
import json, sys
with open('$jsonl_path') as f:
    for line in f:
        issue = json.loads(line)
        if issue.get('id') == '$issue_id':
            print(issue.get('title', ''))
            break
" 2>/dev/null)
    fi
  fi
  issue_title="${issue_title:-Issue $issue_id}"

  issue_desc=$(bd_cmd show "$issue_id" --json 2>/dev/null | parse_bdh_json field description 300 2>/dev/null)
  if [[ -z "$issue_desc" ]]; then
    local jsonl_path="${REPO_ROOT}/.beads/backup/issues.jsonl"
    if [[ -f "$jsonl_path" ]]; then
      issue_desc=$(python3 -c "
import json, sys
with open('$jsonl_path') as f:
    for line in f:
        issue = json.loads(line)
        if issue.get('id') == '$issue_id':
            desc = issue.get('description', '')
            print(desc[:300] if desc else '')
            break
" 2>/dev/null)
    fi
  fi
  if [[ -n "$issue_desc" ]]; then
    issue_objective="${issue_title}. ${issue_desc}"
  else
    issue_objective="$issue_title"
  fi
  issue_args=(--issue "$issue_id" --objective "$issue_objective" "${EXTRA_SWARM_ARGS[@]}")

  log "  [run $run_num] Starting issue=$issue_id log=$run_log"

  run_start=$(date +%s)
  set +e
  (
    cd "$REPO_ROOT"
    bash scripts/run-swarm.sh "${issue_args[@]}" 2>&1
  ) > "$run_log" 2>&1
  exit_code=$?
  set -e
  run_end=$(date +%s)
  elapsed=$((run_end - run_start))

  if [[ $exit_code -eq 0 ]]; then
    log "  [run $run_num] SUCCESS issue=$issue_id (${elapsed}s)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

    # Self-improvement: after successful merge, scan for new issues.
    # The fix may have unmasked new lint/type violations or created
    # opportunities for further improvement.
    if [[ -n "$TARGET_REPO" && -x "$REPO_ROOT/scripts/generate-issues.sh" ]]; then
      log "  [run $run_num] Post-merge: scanning for new issues..."
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
  log "Interrupted. Killing ${#CHILD_PIDS[@]} child processes..."
  for pid in "${CHILD_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  log "Runs=$RUN_COUNT Success=$SUCCESS_COUNT Fail=$FAIL_COUNT"
  exit 130
}
trap cleanup INT TERM

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
          # Discover new issues from bdh ready
          log "Issue list exhausted — discovering new issues from bdh ready..."
          mapfile -t NEW_ISSUES < <(bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null || true)
          if [[ ${#NEW_ISSUES[@]} -gt 0 ]]; then
            ISSUES+=("${NEW_ISSUES[@]}")
            log "  Discovered ${#NEW_ISSUES[@]} new issues: ${NEW_ISSUES[*]}"
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

      # Fallback: bash-side circuit breaker (if pick-next unavailable or returned nothing)
      if [[ -z "$ISSUE_ID" ]]; then
        while IFS= read -r candidate; do
          [[ -z "$candidate" ]] && continue
          if issue_is_exhausted "$candidate"; then
            defer_exhausted_issue "$candidate" "$MAX_ISSUE_FAILURES"
            continue
          fi
          ISSUE_ID="$candidate"
          break
        done < <(bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null || true)
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

    log "  Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  done
else
  # --- Parallel mode ---
  log "=== Parallel mode: up to $PARALLEL concurrent issues ==="

  # Build the issue queue
  if [[ ${#ISSUES[@]} -eq 0 ]]; then
    # Auto mode: fetch ready issues from beads
    mapfile -t ISSUES < <(bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null || true)
    if [[ ${#ISSUES[@]} -eq 0 ]]; then
      log "No ready issues found. Stopping."
      exit 0
    fi
    log "  Found ${#ISSUES[@]} ready issues: ${ISSUES[*]}"
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
      log "Issue list exhausted — discovering new issues from bdh ready..."
      mapfile -t NEW_ISSUES < <(bd_cmd ready --json 2>/dev/null | parse_bdh_json ids 2>/dev/null || true)
      if [[ ${#NEW_ISSUES[@]} -gt 0 ]]; then
        ISSUES+=("${NEW_ISSUES[@]}")
        log "  Discovered ${#NEW_ISSUES[@]} new issues: ${NEW_ISSUES[*]}"
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

    log "=== Batch starting: ${BATCH_SIZE} issues (${IDX}..$(( IDX + BATCH_SIZE - 1 )) of ${#ISSUES[@]}) ==="

    CHILD_PIDS=()
    BATCH_ISSUES=()
    for (( i=0; i<BATCH_SIZE; i++ )); do
      ISSUE_IDX=$((IDX + i))
      ISSUE_ID="${ISSUES[$ISSUE_IDX]}"
      RUN_COUNT=$((RUN_COUNT + 1))
      BATCH_ISSUES+=("$ISSUE_ID")
      run_issue "$RUN_COUNT" "$ISSUE_ID" &
      CHILD_PIDS+=($!)
    done

    # Wait for all in this batch
    for (( i=0; i<${#CHILD_PIDS[@]}; i++ )); do
      pid="${CHILD_PIDS[$i]}"
      issue="${BATCH_ISSUES[$i]}"
      if wait "$pid" 2>/dev/null; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
    done
    CHILD_PIDS=()

    IDX=$((IDX + BATCH_SIZE))

    # Cooldown between batches
    log "  Batch complete. Cooling down ${COOLDOWN}s..."
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
