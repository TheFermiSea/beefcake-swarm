#!/usr/bin/env bash
# reconcile-issues.sh — Fix zombie issues left behind by crashed swarm runs.
#
# Three reconciliation passes:
#   1. Close in_progress issues that have a merged PR (swarm succeeded but didn't close beads)
#   2. Reset in_progress issues with no active worktree or branch (swarm crashed mid-run)
#   3. Clean orphaned worktrees in /tmp/beefcake-wt/ whose issues are closed
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

# Use bd-safe.sh if available (handles shared-server mode), fall back to bd
if [[ -x "$SCRIPT_DIR/bd-safe.sh" ]]; then
  BD_BIN="$SCRIPT_DIR/bd-safe.sh"
else
  BD_BIN="${SWARM_BEADS_BIN:-bd}"
fi
DRY_RUN=0
TARGET_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      if [[ $# -lt 2 ]]; then echo "Error: --repo-root requires a path argument"; exit 1; fi
      TARGET_REPO="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    *)           echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RUN_DIR="${TARGET_REPO:-$REPO_ROOT}"
cd "$RUN_DIR"

log() { echo "[reconcile] $(date +%H:%M:%S) $*"; }

bd_cmd() { "$BD_BIN" "$@" 2>/dev/null; }

# Helper: extract issue IDs from `bd list --json` output.
# Handles both array-of-objects and {issues: [...]} formats.
extract_issue_ids() {
  python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    items = data if isinstance(data, list) else data.get('issues', [])
    for item in items:
        issue_id = item.get('id', '')
        if issue_id:
            print(issue_id)
except: pass
" 2>/dev/null || true
}

# ─── Pass 1: Close issues with merged PRs ────────────────────────────────
#
# When the swarm merges a PR but crashes before closing the beads issue,
# the issue stays in_progress with completed work. We detect this by
# checking gh pr list for merged PRs referencing each issue ID.

reconcile_merged_prs() {
  log "Pass 1: checking for merged PRs with unclosed issues..."

  local closed=0
  local issue_ids

  issue_ids=$(bd_cmd list --status=in_progress --json 2>/dev/null | extract_issue_ids)

  if [[ -z "$issue_ids" ]]; then
    log "  No in_progress issues found."
    return 0
  fi

  local count
  count=$(echo "$issue_ids" | wc -l | tr -d ' ')
  log "  Found $count in_progress issues to check against merged PRs."

  while IFS= read -r issue_id; do
    [[ -z "$issue_id" ]] && continue

    # Search for merged PRs with this exact issue ID in the title.
    # Using "in:title" restricts the search to PR titles only, avoiding false
    # positives from body text matches on short IDs.
    local merged_pr
    merged_pr=$(gh pr list --search "$issue_id in:title" --state merged --json number,title --limit 1 2>/dev/null || echo "[]")

    if [[ "$merged_pr" != "[]" && -n "$merged_pr" ]]; then
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
  done <<< "$issue_ids"

  log "  Pass 1 complete: $closed issues closed."
}

# ─── Pass 2: Reset stale in_progress issues ───────────────────────────────
#
# Issues stuck in_progress with no active worktree or swarm branch are
# zombies from crashed runs. Reset them to open so the swarm can retry.

reconcile_stale_in_progress() {
  log "Pass 2: checking for stale in_progress issues with no active work..."

  local reset_count=0
  local issue_ids

  issue_ids=$(bd_cmd list --status=in_progress --json 2>/dev/null | extract_issue_ids)

  if [[ -z "$issue_ids" ]]; then
    log "  No stale in_progress issues remaining."
    return 0
  fi

  while IFS= read -r issue_id; do
    [[ -z "$issue_id" ]] && continue

    local has_worktree=0
    local has_branch=0

    # Check for an active worktree containing this issue ID (exact match on directory name)
    if [[ -d "/tmp/beefcake-wt/$issue_id" ]]; then
      has_worktree=1
    fi
    # Also check git worktree list for any worktree with this exact issue ID in its path
    if git worktree list 2>/dev/null | grep -qF "/$issue_id "; then
      has_worktree=1
    fi

    # Check for a swarm branch (exact match)
    if git rev-parse --verify "swarm/$issue_id" >/dev/null 2>&1; then
      has_branch=1
    fi

    if [[ "$has_worktree" -eq 0 && "$has_branch" -eq 0 ]]; then
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN: would reset $issue_id to open (no worktree, no branch)"
      else
        log "  Resetting $issue_id to open (no worktree, no branch)"
        bd_cmd update "$issue_id" --status=open 2>/dev/null || true
        reset_count=$((reset_count + 1))
      fi
    else
      log "  Skipping $issue_id — active work (worktree=$has_worktree, branch=$has_branch)"
    fi
  done <<< "$issue_ids"

  log "  Pass 2 complete: $reset_count issues reset to open."
}

# ─── Pass 3: Clean orphaned worktrees ─────────────────────────────────────
#
# Worktrees in /tmp/beefcake-wt/ whose issue is already closed are orphans.
# Only removes worktrees when we can positively confirm the issue is closed —
# "unknown" status (bd server down, parse error) is treated as "keep".

reconcile_orphaned_worktrees() {
  log "Pass 3: checking for orphaned worktrees..."

  local cleaned=0
  local wt_base="/tmp/beefcake-wt"

  [[ -d "$wt_base" ]] || { log "  No worktree directory at $wt_base."; return 0; }

  for wt_dir in "$wt_base"/*/; do
    [[ -d "$wt_dir" ]] || continue
    local issue_id
    issue_id=$(basename "$wt_dir")

    # Check issue status — only remove if positively confirmed closed
    local status
    status=$(bd_cmd show "$issue_id" --json 2>/dev/null \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null \
      || echo "")

    if [[ "$status" == "closed" ]]; then
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN: would remove orphaned worktree $wt_dir (issue closed)"
      else
        log "  Removing orphaned worktree $wt_dir (issue closed)"
        rm -rf "$wt_dir"
        cleaned=$((cleaned + 1))
      fi
    elif [[ -z "$status" || "$status" == "unknown" ]]; then
      log "  Skipping $wt_dir — could not determine issue status (keeping)"
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
