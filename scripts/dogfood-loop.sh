#!/usr/bin/env bash
# dogfood-loop.sh — Repeatedly run swarm-agents to work through beads issues.
#
# Usage:
#   ./scripts/dogfood-loop.sh                    # Pick from bd ready (default)
#   ./scripts/dogfood-loop.sh --max-runs 5       # Stop after 5 runs
#   ./scripts/dogfood-loop.sh --cooldown 120     # 2-minute cooldown between runs
#   ./scripts/dogfood-loop.sh --issue-list "beefcake-w70b.3.5.1 beefcake-w70b.6.4.2"
#                                                # Work specific issues in order
#
# Environment:
#   SWARM_CLOUD_API_KEY    Required (passed through to run-swarm.sh)
#   DOGFOOD_LOG_DIR        Log directory (default: ./logs/dogfood)
#   DOGFOOD_MAX_RUNS       Max iterations (default: unlimited)
#   DOGFOOD_COOLDOWN       Seconds between runs (default: 60)
#   DOGFOOD_ISSUE_LIST     Space-separated issue IDs to work in order
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
MAX_RUNS="${DOGFOOD_MAX_RUNS:-0}"       # 0 = unlimited
COOLDOWN="${DOGFOOD_COOLDOWN:-60}"
LOG_DIR="${DOGFOOD_LOG_DIR:-${REPO_ROOT}/logs/dogfood}"
ISSUE_LIST="${DOGFOOD_ISSUE_LIST:-}"

# CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-runs)    MAX_RUNS="$2"; shift 2 ;;
    --cooldown)    COOLDOWN="$2"; shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --issue-list)  ISSUE_LIST="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# --- Telemetry ---
DOGFOOD_START=$(date +%s)
RUN_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SUMMARY_FILE="${LOG_DIR}/dogfood-summary.jsonl"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

record_run() {
  local run_num="$1" issue_id="$2" exit_code="$3" elapsed="$4" log_file="$5"
  printf '{"run":%d,"issue":"%s","exit_code":%d,"elapsed_s":%d,"timestamp":"%s","log":"%s"}\n' \
    "$run_num" "$issue_id" "$exit_code" "$elapsed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$log_file" \
    >> "$SUMMARY_FILE"
}

# --- Pre-flight ---
log "Dogfood loop starting"
log "  Repo:     $REPO_ROOT"
log "  Logs:     $LOG_DIR"
log "  Cooldown: ${COOLDOWN}s"
log "  Max runs: $([ "$MAX_RUNS" -eq 0 ] && echo 'unlimited' || echo "$MAX_RUNS")"

if [[ -n "$ISSUE_LIST" ]]; then
  read -ra ISSUES <<< "$ISSUE_LIST"
  log "  Issues:   ${ISSUES[*]} (${#ISSUES[@]} total)"
else
  ISSUES=()
  log "  Issues:   auto (from bd ready)"
fi

# Verify toolchain
if ! command -v cargo &>/dev/null; then
  echo "ERROR: cargo not found. Install Rust toolchain first." >&2
  exit 1
fi
if ! command -v bd &>/dev/null; then
  echo "WARNING: bd not found. Swarm will use NoOpTracker." >&2
fi

# --- Main loop ---
trap 'log "Interrupted. Runs=$RUN_COUNT Success=$SUCCESS_COUNT Fail=$FAIL_COUNT"; exit 130' INT TERM

while true; do
  RUN_COUNT=$((RUN_COUNT + 1))

  # Check max runs
  if [[ "$MAX_RUNS" -gt 0 && "$RUN_COUNT" -gt "$MAX_RUNS" ]]; then
    log "Reached max runs ($MAX_RUNS). Stopping."
    break
  fi

  # Determine which issue to work
  ISSUE_ARGS=()
  ISSUE_ID="auto"
  if [[ ${#ISSUES[@]} -gt 0 ]]; then
    IDX=$((RUN_COUNT - 1))
    if [[ $IDX -ge ${#ISSUES[@]} ]]; then
      log "All ${#ISSUES[@]} issues processed. Stopping."
      break
    fi
    ISSUE_ID="${ISSUES[$IDX]}"
    # Fetch title from beads for the objective
    ISSUE_TITLE=$(bd show "$ISSUE_ID" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)[0].get('title',''))" 2>/dev/null || echo "Issue $ISSUE_ID")
    ISSUE_ARGS=(--issue "$ISSUE_ID" --objective "$ISSUE_TITLE")
  fi

  RUN_LOG="${LOG_DIR}/run-${RUN_COUNT}-${ISSUE_ID}-$(date +%Y%m%d-%H%M%S).log"
  log "=== Run $RUN_COUNT: issue=$ISSUE_ID ==="
  log "  Log: $RUN_LOG"

  RUN_START=$(date +%s)
  set +e
  (
    cd "$REPO_ROOT"
    bash scripts/run-swarm.sh "${ISSUE_ARGS[@]}" 2>&1
  ) > "$RUN_LOG" 2>&1
  EXIT_CODE=$?
  set -e
  RUN_END=$(date +%s)
  ELAPSED=$((RUN_END - RUN_START))

  if [[ $EXIT_CODE -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    log "  Result: SUCCESS (${ELAPSED}s)"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    log "  Result: FAILED exit=$EXIT_CODE (${ELAPSED}s)"
    # Show last 5 lines of log for quick diagnosis
    log "  Tail:"
    tail -5 "$RUN_LOG" | while IFS= read -r line; do
      log "    $line"
    done
  fi

  record_run "$RUN_COUNT" "$ISSUE_ID" "$EXIT_CODE" "$ELAPSED" "$RUN_LOG"

  # Check if there are more issues to work (auto mode)
  if [[ ${#ISSUES[@]} -eq 0 ]]; then
    READY_COUNT=$(bd ready 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [[ "$READY_COUNT" -eq 0 ]]; then
      log "No more ready issues. Stopping."
      break
    fi
    log "  $READY_COUNT issues remaining in bd ready"
  fi

  # Cooldown (unless this is the last run)
  if [[ "$MAX_RUNS" -gt 0 && "$RUN_COUNT" -ge "$MAX_RUNS" ]]; then
    break
  fi
  if [[ ${#ISSUES[@]} -gt 0 && $RUN_COUNT -ge ${#ISSUES[@]} ]]; then
    break
  fi

  log "  Cooling down ${COOLDOWN}s..."
  sleep "$COOLDOWN"
done

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
