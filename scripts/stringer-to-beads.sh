#!/usr/bin/env bash
# stringer-to-beads.sh — Import a Stringer JSONL scan into the beads backlog.
#
# Reads a Stringer scan file (one JSON record per line) and creates beads
# issues for signals not yet imported. Dedup is tracked via a local state
# file so repeated runs are safe and fast.
#
# Usage:
#   scripts/stringer-to-beads.sh [SCAN_FILE] [OPTIONS]
#
# Options:
#   --max-priority N     Only import signals with priority <= N (default: 3)
#                        Priority 4 = pure duplication noise; skip by default.
#   --max-new N          Cap new issues created per run (default: 30)
#   --dry-run            Print what would be created; do not call bd create.
#   --label-filter LABEL Only import signals that have LABEL in their labels.
#                        Can be specified multiple times.
#   --include-all        Override --max-priority; import all priorities.
#   --state-file PATH    Path to dedup state file (default: docs/stringer/.imported-ids)
#
# Dedup:
#   Stringer IDs (e.g. str-coordination-21eca9cb) are stored one-per-line in
#   STATE_FILE after each successful bd create.  On subsequent runs the file is
#   checked before creating, making repeated imports idempotent.
#
# Integration with dogfood-loop --discover:
#   Run this script before starting a discover loop to seed the ready queue:
#     scripts/stringer-to-beads.sh && dogfood-loop.sh --discover ...

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_STATE="$REPO_ROOT/docs/stringer/.imported-ids"

# ── Exclusive lock — prevent concurrent imports corrupting the state file ──────
LOCK_FILE="${TMPDIR:-/tmp}/stringer-to-beads.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[stringer-to-beads] Another instance is running (lock: $LOCK_FILE). Exiting." >&2
  exit 0
fi

SCAN_FILE=""
MAX_PRIORITY=3
MAX_NEW=30
DRY_RUN=0
STATE_FILE="$DEFAULT_STATE"
INCLUDE_ALL=0
LABEL_FILTERS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-priority)   MAX_PRIORITY="$2"; shift 2 ;;
    --max-new)        MAX_NEW="$2";       shift 2 ;;
    --dry-run)        DRY_RUN=1;          shift   ;;
    --label-filter)   LABEL_FILTERS+=("$2"); shift 2 ;;
    --include-all)    INCLUDE_ALL=1;      shift   ;;
    --state-file)     STATE_FILE="$2";    shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  SCAN_FILE="$1"; shift ;;
  esac
done

if [[ -z "$SCAN_FILE" ]]; then
  # Auto-detect: most recent scan in docs/stringer/
  SCAN_FILE=$(ls -t "$REPO_ROOT"/docs/stringer/scan-*.jsonl 2>/dev/null | head -1 || true)
  if [[ -z "$SCAN_FILE" ]]; then
    echo "ERROR: No scan file found in docs/stringer/ and none specified." >&2
    exit 1
  fi
fi

[[ -f "$SCAN_FILE" ]] || { echo "ERROR: Scan file not found: $SCAN_FILE" >&2; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '[stringer-to-beads] %s\n' "$1"; }
info() { printf '  %s\n' "$1"; }

bd_cmd() { command -v bd &>/dev/null && bd "$@" || bdh "$@"; }

already_imported() {
  local id="$1"
  [[ -f "$STATE_FILE" ]] && grep -qxF "$id" "$STATE_FILE"
}

mark_imported() {
  local id="$1"
  mkdir -p "$(dirname "$STATE_FILE")"
  echo "$id" >> "$STATE_FILE"
}

# ── Main import loop ──────────────────────────────────────────────────────────
log "Stringer → Beads import"
log "  Scan:         $SCAN_FILE"
log "  State file:   $STATE_FILE"
log "  Max priority: $([ "$INCLUDE_ALL" -eq 1 ] && echo 'all' || echo "P$MAX_PRIORITY")"
log "  Max new:      $MAX_NEW"
log "  Dry run:      $([ "$DRY_RUN" -eq 1 ] && echo YES || echo no)"
[[ ${#LABEL_FILTERS[@]} -gt 0 ]] && log "  Label filter: ${LABEL_FILTERS[*]}"
echo

CREATED=0
SKIPPED_DUP=0
SKIPPED_PRIO=0
SKIPPED_LABEL=0
TOTAL=0

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  TOTAL=$((TOTAL + 1))

  # Parse fields with python for safety (jq may not be installed).
  read -r str_id title description priority labels_json <<< "$(python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
labels = json.dumps(r.get('labels', []))
print(r['id'], '|||', r['title'], '|||', r.get('description',''), '|||', r['priority'], '|||', labels)
" <<< "$line" | awk -F' \\|\\|\\| ' '{print $1, "###", $2, "###", $3, "###", $4, "###", $5}')"

  # Re-parse cleanly with python into env vars to handle spaces/special chars.
  eval "$(python3 - "$line" <<'PYEOF'
import json, sys, shlex
r = json.loads(sys.argv[1])
print("str_id=" + shlex.quote(r["id"]))
print("title=" + shlex.quote(r["title"]))
print("description=" + shlex.quote(r.get("description", "")))
print("priority=" + shlex.quote(str(r["priority"])))
# labels as space-separated for bash
label_str = " ".join(r.get("labels", []))
print("labels_raw=" + shlex.quote(label_str))
PYEOF
)"

  # ── Filter: max-priority ──
  if [[ "$INCLUDE_ALL" -eq 0 && "$priority" -gt "$MAX_PRIORITY" ]]; then
    SKIPPED_PRIO=$((SKIPPED_PRIO + 1))
    continue
  fi

  # ── Filter: label-filter ──
  if [[ ${#LABEL_FILTERS[@]} -gt 0 ]]; then
    match=0
    for lf in "${LABEL_FILTERS[@]}"; do
      if [[ " $labels_raw " == *" $lf "* ]]; then match=1; break; fi
    done
    if [[ $match -eq 0 ]]; then
      SKIPPED_LABEL=$((SKIPPED_LABEL + 1))
      continue
    fi
  fi

  # ── Dedup check ──
  if already_imported "$str_id"; then
    SKIPPED_DUP=$((SKIPPED_DUP + 1))
    continue
  fi

  # ── Cap ──
  if [[ $CREATED -ge $MAX_NEW ]]; then
    log "Reached --max-new=$MAX_NEW cap. Stopping early."
    break
  fi

  # ── Build bd create labels ──
  # Keep stringer labels useful for the swarm + add stringer-id for dedup.
  # Strip very noisy/redundant labels that don't help the agent.
  KEEP_LABELS=("stringer-generated" "stringer-id:${str_id}")
  for lbl in $labels_raw; do
    case "$lbl" in
      stringer-generated) ;;  # already added
      workspace:*) KEEP_LABELS+=("$lbl") ;;
      vuln|security|complexity|dead-code|deadcode|duplication|churn|gitlog) KEEP_LABELS+=("$lbl") ;;
      CVE-*|RUSTSEC-*|GHSA-*) KEEP_LABELS+=("$lbl") ;;
      missing-tests|low-test-ratio|large-file|coupling|todo|todos) KEEP_LABELS+=("$lbl") ;;
      # skip: near-clone, code-clone, refactor-candidate, cleanup-candidate, patterns, gitlog (dup with churn)
    esac
  done
  LABEL_ARG=$(IFS=,; echo "${KEEP_LABELS[*]}")

  # ── Create or dry-run ──
  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "[DRY-RUN] Would create P$priority: $title"
    info "          labels: $LABEL_ARG"
    info "          description: ${description:0:120}..."
    CREATED=$((CREATED + 1))
  else
    if bd_cmd create \
        --title "$title" \
        --description "$description" \
        --priority "$priority" \
        --type task \
        --labels "$LABEL_ARG" \
        2>/dev/null; then
      mark_imported "$str_id"
      CREATED=$((CREATED + 1))
      info "Created P$priority: $title"
    else
      echo "WARN: bd create failed for $str_id" >&2
    fi
  fi

done < "$SCAN_FILE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
log "Done."
log "  Total records:   $TOTAL"
log "  Created:         $CREATED"
log "  Skipped (dedup): $SKIPPED_DUP"
log "  Skipped (prio):  $SKIPPED_PRIO"
log "  Skipped (label): $SKIPPED_LABEL"
if [[ "$DRY_RUN" -eq 0 && "$CREATED" -gt 0 ]]; then
  log "  Sync:            run 'bd dolt push' to replicate to remote"
fi
