#!/usr/bin/env bash
# lint-janitor.sh — Auto-fix trivial lint/format/type issues without LLM delegation.
#
# Replaces generate-issues.sh in the post-merge step of dogfood-loop.sh.
# Instead of creating beads issues for every lint finding (which wastes
# expensive swarm cycles), this script:
#
#   Tier 0: Deterministic auto-fix (black, ruff --fix, type stubs install)
#   Tier 1: Only report findings that CANNOT be auto-fixed
#   Tier 2: Create a beads issue only if non-trivial AND no duplicate exists
#
# Usage:
#   ./scripts/lint-janitor.sh <repo-root>
#   MAX_ISSUES=3 ./scripts/lint-janitor.sh ~/code/CF-LIBS-improved
#
# Env vars:
#   MAX_ISSUES      — max beads issues to create for non-fixable findings (default: 3)
#   JANITOR_DRY_RUN — if set to 1, don't commit or create issues (default: 0)
#   BD              — beads CLI binary (default: bd)
set -euo pipefail

REPO_ROOT="${1:?Usage: lint-janitor.sh <repo-root>}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BD="${SWARM_BEADS_BIN:-${BD:-bd}}"
MAX_ISSUES="${MAX_ISSUES:-3}"
DRY_RUN="${JANITOR_DRY_RUN:-0}"

log() { echo "[lint-janitor $(date -Iseconds)] $*"; }

cd "$REPO_ROOT"

# Detect language from profile or directory structure
LANG="unknown"
if [[ -f .swarm/profile.toml ]]; then
  LANG=$(grep -oP 'language\s*=\s*"\K[^"]+' .swarm/profile.toml 2>/dev/null || echo "unknown")
fi
[[ "$LANG" == "unknown" && -d cflibs ]] && LANG="python"
[[ "$LANG" == "unknown" && -f Cargo.toml ]] && LANG="rust"

fixes_applied=0
issues_created=0

# ============================================================
# TIER 0: Deterministic auto-fix (zero cost)
# ============================================================

if [[ "$LANG" == "python" ]]; then
  src_dir="cflibs"
  [[ -d cflibs ]] || src_dir="src"

  # --- Black formatting ---
  if command -v black &>/dev/null; then
    log "Tier 0: running black on $src_dir/"
    if black "$src_dir/" --quiet 2>/dev/null; then
      if ! git diff --quiet 2>/dev/null; then
        fixes_applied=$((fixes_applied + 1))
        log "  black: formatted files"
      fi
    fi
  fi

  # --- Ruff auto-fix ---
  if command -v ruff &>/dev/null; then
    log "Tier 0: running ruff --fix on $src_dir/"
    if ruff check "$src_dir/" --fix --quiet 2>/dev/null; then
      if ! git diff --quiet 2>/dev/null; then
        fixes_applied=$((fixes_applied + 1))
        log "  ruff: auto-fixed violations"
      fi
    fi
  fi

  # --- Mypy type stubs installation ---
  # This is the root cause of the duplicate yaml stubs issue.
  # Install missing stubs into the venv before mypy runs.
  if command -v mypy &>/dev/null; then
    log "Tier 0: checking for missing mypy type stubs"
    # Activate venv if available
    venv_activate=""
    for v in .venv/bin/activate "$REPO_ROOT/.venv/bin/activate"; do
      if [[ -f "$v" ]]; then
        venv_activate="$v"
        break
      fi
    done

    if [[ -n "$venv_activate" ]]; then
      # shellcheck disable=SC1090
      source "$venv_activate"
      # Install common missing stubs non-interactively
      mypy_output=$(mypy "$src_dir/" --show-error-codes --no-error-summary 2>/dev/null || true)
      missing_stubs=$(echo "$mypy_output" | grep -oP 'Library stubs not installed for "\K[^"]+' | sort -u || true)
      if [[ -n "$missing_stubs" ]]; then
        for stub in $missing_stubs; do
          stub_pkg="types-${stub}"
          log "  Installing missing stub: $stub_pkg"
          pip install "$stub_pkg" --quiet 2>/dev/null || true
        done
        fixes_applied=$((fixes_applied + 1))
      fi
    fi
  fi

elif [[ "$LANG" == "rust" ]]; then
  # --- Cargo fmt ---
  log "Tier 0: running cargo fmt"
  cargo fmt --all 2>/dev/null || true
  if ! git diff --quiet 2>/dev/null; then
    fixes_applied=$((fixes_applied + 1))
    log "  cargo fmt: formatted files"
  fi

  # --- Cargo clippy --fix ---
  log "Tier 0: running cargo clippy --fix"
  cargo clippy --fix --allow-dirty --allow-staged -- -D warnings 2>/dev/null || true
  if ! git diff --quiet 2>/dev/null; then
    fixes_applied=$((fixes_applied + 1))
    log "  clippy: auto-fixed warnings"
  fi
fi

# Commit Tier 0 fixes if any
if [[ $fixes_applied -gt 0 ]]; then
  if [[ "$DRY_RUN" -eq 0 ]]; then
    git add -A 2>/dev/null || true
    git commit -m "swarm: lint-janitor auto-fix ($fixes_applied tool passes)" --quiet 2>/dev/null || true
    log "Tier 0: committed $fixes_applied auto-fix passes"
  else
    log "Tier 0: would commit $fixes_applied auto-fix passes (dry-run)"
  fi
fi

# ============================================================
# TIER 1: Identify remaining non-auto-fixable findings
# ============================================================

remaining_findings=()

if [[ "$LANG" == "python" ]]; then
  # Re-run mypy AFTER stub installation to see what's actually broken
  if command -v mypy &>/dev/null; then
    log "Tier 1: re-running mypy to find non-auto-fixable issues"
    # Use venv if available
    if [[ -n "${venv_activate:-}" ]]; then
      # shellcheck disable=SC1090
      source "$venv_activate"
    fi
    mypy_output=$(mypy "$src_dir/" --show-error-codes --no-error-summary 2>/dev/null || true)
    mypy_errors=$(echo "$mypy_output" | grep ": error:" | grep -v "import-untyped" || true)
    if [[ -n "$mypy_errors" ]]; then
      # Group by file+code, take unique signatures
      while IFS= read -r sig; do
        [[ -z "$sig" ]] && continue
        remaining_findings+=("mypy: $sig")
      done < <(echo "$mypy_errors" | sed 's/:[0-9]*:/:/' | sort -u | head -"$MAX_ISSUES")
    fi
  fi

  # Re-run ruff for unfixable violations
  if command -v ruff &>/dev/null; then
    ruff_remaining=$(ruff check "$src_dir/" --output-format text 2>/dev/null || true)
    if [[ -n "$ruff_remaining" ]]; then
      ruff_count=$(echo "$ruff_remaining" | grep -c ":" || echo 0)
      if [[ "$ruff_count" -gt 0 ]]; then
        remaining_findings+=("ruff: $ruff_count unfixable violations remaining")
      fi
    fi
  fi
fi

if [[ ${#remaining_findings[@]} -eq 0 ]]; then
  log "All clean — no non-auto-fixable findings."
  log "Done. $fixes_applied auto-fixes applied, $issues_created issues created."
  exit 0
fi

log "Tier 1: ${#remaining_findings[@]} non-auto-fixable finding(s) remain"

# ============================================================
# TIER 2: Create beads issues ONLY for non-trivial, non-duplicate findings
# ============================================================

# Fetch existing open issue titles for dedup
existing_titles=""
if [[ "$DRY_RUN" -eq 0 ]]; then
  existing_titles=$("$BD" list --status=open --limit 0 2>/dev/null || true)
  existing_titles+=$'\n'
  existing_titles+=$("$BD" list --status=in_progress --limit 0 2>/dev/null || true)
fi

for finding in "${remaining_findings[@]}"; do
  if [[ $issues_created -ge $MAX_ISSUES ]]; then
    log "Tier 2: reached max issues cap ($MAX_ISSUES), stopping"
    break
  fi

  # Skip import-untyped — should have been fixed by stub installation
  if echo "$finding" | grep -q "import-untyped"; then
    log "  Skipping import-untyped (should be resolved by stub install)"
    continue
  fi

  # Extract a short title from the finding
  title=$(echo "$finding" | head -1 | cut -c1-120)

  # Check for duplicates in existing issues
  title_lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')
  if echo "$existing_titles" | tr '[:upper:]' '[:lower:]' | grep -qF "$title_lower"; then
    log "  Skipping duplicate: $title"
    continue
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$BD" create \
      --title="[lint] $title" \
      --description="Non-auto-fixable finding from lint-janitor post-merge scan. Manual fix required." \
      --type=bug --priority=3 2>/dev/null || true
    issues_created=$((issues_created + 1))
    log "  Created issue: [lint] $title"
  else
    log "  Would create issue: [lint] $title (dry-run)"
    issues_created=$((issues_created + 1))
  fi
done

log "Done. $fixes_applied auto-fixes applied, $issues_created issues created."
