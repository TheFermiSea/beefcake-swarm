#!/usr/bin/env bash
# reconcile-issues.sh — Fix zombie issues left behind by crashed swarm runs.
#
# Two reconciliation passes:
#   1. Close in_progress/open issues that have a merged PR (swarm succeeded but didn't close beads)
#   2. Reset in_progress issues with no active worktree or branch (swarm crashed mid-run)
#
# Safe to run repeatedly — idempotent. Called from dogfood-loop.sh between batches
# and can be run standalone for manual cleanup.
#
# Usage:
#   ./scripts/reconcile-issues.sh                  # Reconcile current repo
#   ./scripts/reconcile-issues.sh --repo-root /path # Reconcile a different repo
#   ./scripts/reconcile-issues.sh --dry-run         # Show what would change
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BD_BIN="${SWARM_BEADS_BIN:-bd}"
DRY_RUN=0
TARGET_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) TARGET_REPO="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    *)           echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RUN_DIR="${TARGET_REPO:-$REPO_ROOT}"
cd "$RUN_DIR"

log() { echo "[reconcile] $(date +%H:%M:%S) $*"; }

bd_cmd() { "$BD_BIN" "$@" 2>/dev/null; }

# ─── Pass 1: Close issues with merged PRs ────────────────────────────────
#
# When the swarm merges a PR but crashes before closing the beads issue,
# the issue stays in_progress or open with completed work. We detect this
# by checking gh pr list for merged PRs referencing each issue ID.

reconcile_merged_prs() {
  log "Pass 1: checking for merged PRs with unclosed issues..."

  local closed=0
  local in_progress_ids

  # Get all in_progress issues
  in_progress_ids=$(bd_cmd list --status=in_progress --json 2>/dev/null \
    | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for item in data:
            issue_id = item.get('id', '')
            if issue_id:
                print(issue_id)
    elif isinstance(data, dict) and 'issues' in data:
        for item in data['issues']:
            issue_id = item.get('id', '')
            if issue_id:
                print(issue_id)
except: pass
" 2>/dev/null || true)

  if [[ -z "$in_progress_ids" ]]; then
    log "  No in_progress issues found."
    return 0
  fi

  local count
  count=$(echo "$in_progress_ids" | wc -l | tr -d ' ')
  log "  Found $count in_progress issues to check."

  while IFS= read -r issue_id; do
    [[ -z "$issue_id" ]] && continue

    # Search for merged PRs mentioning this issue ID in title or body
    local merged_pr
    merged_pr=$(gh pr list --search "$issue_id" --state merged --json number,title --limit 1 2>/dev/null || echo "[]")

    if [[ "$merged_pr" != "[]" && "$merged_pr" != "" ]]; then
      local pr_num pr_title
      pr_num=$(echo "$merged_pr" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['number'] if d else '')" 2>/dev/null || echo "")
      pr_title=$(echo "$merged_pr" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['title'][:80] if d else '')" 2>/dev/null || echo "")

      if [[ -n "$pr_num" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
          log "  DRY-RUN: would close $issue_id (merged PR #$pr_num: $pr_title)"
        else
          log "  Closing $issue_id — merged PR #$pr_num: $pr_title"
          bd_cmd close "$issue_id" --reason "Auto-reconciled: PR #$pr_num merged ($pr_title)" --force 2>/dev/null || true
          closed=$((closed + 1))
        fi
      fi
    fi
  done <<< "$in_progress_ids"

  log "  Pass 1 complete: $closed issues closed."
}

# ─── Pass 2: Reset stale in_progress issues ───────────────────────────────
#
# Issues stuck in_progress with no active worktree or swarm branch are
# zombies from crashed runs. Reset them to open so the swarm can retry.

reconcile_stale_in_progress() {
  log "Pass 2: checking for stale in_progress issues with no active work..."

  local reset=0
  local in_progress_ids

  in_progress_ids=$(bd_cmd list --status=in_progress --json 2>/dev/null \
    | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for item in data:
            issue_id = item.get('id', '')
            if issue_id:
                print(issue_id)
    elif isinstance(data, dict) and 'issues' in data:
        for item in data['issues']:
            issue_id = item.get('id', '')
            if issue_id:
                print(issue_id)
except: pass
" 2>/dev/null || true)

  if [[ -z "$in_progress_ids" ]]; then
    log "  No stale in_progress issues remaining."
    return 0
  fi

  # Get active worktree directories
  local active_worktrees
  active_worktrees=$(git worktree list --porcelain 2>/dev/null | grep "^worktree " | awk '{print $2}' || true)

  while IFS= read -r issue_id; do
    [[ -z "$issue_id" ]] && continue

    local has_worktree=0
    local has_branch=0

    # Check for an active worktree containing this issue ID
    if echo "$active_worktrees" | grep -q "$issue_id" 2>/dev/null; then
      has_worktree=1
    fi
    # Check /tmp/beefcake-wt/<issue_id>
    if [[ -d "/tmp/beefcake-wt/$issue_id" ]]; then
      has_worktree=1
    fi

    # Check for a swarm branch
    if git branch --list "swarm/$issue_id" 2>/dev/null | grep -q .; then
      has_branch=1
    fi

    if [[ "$has_worktree" -eq 0 && "$has_branch" -eq 0 ]]; then
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN: would reset $issue_id to open (no worktree, no branch)"
      else
        log "  Resetting $issue_id to open (no worktree, no branch)"
        bd_cmd update "$issue_id" --status=open 2>/dev/null || true
        reset=$((reset + 1))
      fi
    else
      log "  Skipping $issue_id — active work (worktree=$has_worktree, branch=$has_branch)"
    fi
  done <<< "$in_progress_ids"

  log "  Pass 2 complete: $reset issues reset to open."
}

# ─── Pass 3: Clean orphaned worktrees ─────────────────────────────────────
#
# Worktrees in /tmp/beefcake-wt/ whose issue is already closed are orphans.

reconcile_orphaned_worktrees() {
  log "Pass 3: checking for orphaned worktrees..."

  local cleaned=0
  local wt_base="/tmp/beefcake-wt"

  [[ -d "$wt_base" ]] || { log "  No worktree directory at $wt_base."; return 0; }

  for wt_dir in "$wt_base"/*/; do
    [[ -d "$wt_dir" ]] || continue
    local issue_id
    issue_id=$(basename "$wt_dir")

    # Check if this issue is closed
    local status
    status=$(bd_cmd show "$issue_id" --json 2>/dev/null \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null \
      || echo "unknown")

    if [[ "$status" == "closed" || "$status" == "unknown" ]]; then
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN: would remove orphaned worktree $wt_dir (issue $status)"
      else
        log "  Removing orphaned worktree $wt_dir (issue $status)"
        rm -rf "$wt_dir"
        cleaned=$((cleaned + 1))
      fi
    fi
  done

  # Prune git worktree references
  if [[ "$DRY_RUN" -eq 0 ]]; then
    git worktree prune 2>/dev/null || true
  fi

  log "  Pass 3 complete: $cleaned orphaned worktrees removed."
}

# ─── Main ─────────────────────────────────────────────────────────────────

log "Reconciling issues in $(pwd)"
[[ "$DRY_RUN" -eq 1 ]] && log "  (DRY RUN — no changes will be made)"

reconcile_merged_prs
reconcile_stale_in_progress
reconcile_orphaned_worktrees

log "Done."
