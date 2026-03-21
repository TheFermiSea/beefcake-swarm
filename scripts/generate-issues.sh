#!/usr/bin/env bash
# generate-issues.sh — Auto-generate beads issues from linter/checker output.
#
# Runs quality gate tools on a target repo and creates beads issues from
# each finding. Designed for the beefcake-loop: seed a target repo's beads
# database with actionable improvement tasks.
#
# Usage:
#   ./scripts/generate-issues.sh <repo-root>
#   ./scripts/generate-issues.sh ~/code/CF-LIBS-improved
#
# Requires: bd (beads CLI), ruff, mypy (for Python targets)
# Reads: .swarm/profile.toml for language-specific gates
set -euo pipefail

REPO_ROOT="${1:?Usage: generate-issues.sh <repo-root>}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"  # absolute path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BD="${SWARM_BEADS_BIN:-bd}"
MAX_ISSUES="${MAX_ISSUES:-30}"  # cap to avoid flooding

log() { echo "[generate-issues $(date -Iseconds)] $*"; }

# Verify beads is initialized
if [[ ! -d "$REPO_ROOT/.beads" ]]; then
  log "Initializing beads in $REPO_ROOT"
  (cd "$REPO_ROOT" && "$BD" init 2>/dev/null || true)
fi

cd "$REPO_ROOT"

# Count existing open issues to avoid duplicates
existing=$("$BD" list --status=open 2>/dev/null | wc -l || echo 0)
log "Existing open issues: $existing"

created=0

# --- Ruff (Python linting) ---
if command -v ruff &>/dev/null && [[ -d cflibs || -d src ]]; then
  # Detect source directory
  src_dir="cflibs"
  [[ -d cflibs ]] || src_dir="src"

  log "Running ruff check on $src_dir/ ..."
  ruff_output=$(ruff check "$src_dir/" --output-format json 2>/dev/null || true)

  if [[ -n "$ruff_output" && "$ruff_output" != "[]" ]]; then
    count=$(echo "$ruff_output" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
    log "Ruff found $count violations"

    # Group by file+code to avoid one issue per line
    echo "$ruff_output" | python3 "$SCRIPT_DIR/lint-to-beads.py" \
      --tool ruff --priority 3 --max-issues "$MAX_ISSUES" --bd "$BD" 2>/dev/null || true
    created=$((created + $(echo "$ruff_output" | python3 "$SCRIPT_DIR/lint-to-beads.py" \
      --tool ruff --priority 3 --max-issues "$MAX_ISSUES" --bd "$BD" --dry-run 2>/dev/null | wc -l || echo 0)))
  else
    log "Ruff: clean (no violations)"
  fi
fi

# --- Mypy (Python type checking) ---
if command -v mypy &>/dev/null && [[ -d cflibs || -d src ]]; then
  src_dir="cflibs"
  [[ -d cflibs ]] || src_dir="src"

  log "Running mypy on $src_dir/ ..."
  # mypy outputs to stdout, one error per line: file:line: error: message [code]
  mypy_output=$(mypy "$src_dir/" --show-error-codes --no-error-summary 2>/dev/null || true)

  if [[ -n "$mypy_output" ]]; then
    count=$(echo "$mypy_output" | grep -c ": error:" || echo 0)
    log "Mypy found $count errors"

    echo "$mypy_output" | python3 "$SCRIPT_DIR/mypy-to-beads.py" \
      --priority 2 --max-issues "$MAX_ISSUES" --bd "$BD" 2>/dev/null || true
  else
    log "Mypy: clean (no errors)"
  fi
fi

# --- Missing test coverage (Python) ---
if [[ -d tests && -d cflibs ]]; then
  log "Checking for untested modules..."
  # Find Python modules without corresponding test files
  for module in cflibs/*/*.py; do
    [[ -f "$module" ]] || continue
    basename=$(basename "$module" .py)
    [[ "$basename" == "__init__" ]] && continue
    dirname=$(basename "$(dirname "$module")")

    # Check if a test file exists
    test_file="tests/test_${basename}.py"
    test_file_alt="tests/test_${dirname}_${basename}.py"
    if [[ ! -f "$test_file" && ! -f "$test_file_alt" ]]; then
      # Check if already tracked
      existing_title="Add tests for $dirname/$basename"
      if "$BD" search "$existing_title" 2>/dev/null | grep -q "$existing_title"; then
        continue
      fi

      log "Creating issue: $existing_title"
      "$BD" create \
        --title="$existing_title" \
        --description="Module cflibs/$dirname/$basename.py has no corresponding test file. Create tests/test_${basename}.py with unit tests covering the public API." \
        --type=task --priority=3 2>/dev/null || true
      created=$((created + 1))

      if [[ $created -ge $MAX_ISSUES ]]; then
        log "Reached max issues cap ($MAX_ISSUES)"
        break 2
      fi
    fi
  done
fi

log "Done. Created $created new issues."
log "Run 'cd $REPO_ROOT && $BD ready' to see available work."
