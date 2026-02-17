#!/usr/bin/env bash
# verify-task.sh — TaskCompleted quality gate hook
# Replicates coordination crate's verifier pipeline as shell gates.
# Exit 0 = pass, Exit 2 = reject with feedback to agent.
set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
SUBJECT=$(echo "$INPUT" | jq -r '.task_subject // empty' 2>/dev/null || echo "")

# Skip for non-implementation tasks
if echo "$SUBJECT" | grep -qiE '(research|plan|analyze|investigate|review|document)'; then
  echo "Skipping quality gates for non-implementation task: $SUBJECT"
  exit 0
fi

ERRORS=""

# Gate 1: cargo fmt
echo "Gate 1/4: cargo fmt --check"
if ! cargo fmt --all -- --check 2>&1; then
  ERRORS="${ERRORS}FORMATTING: Code is not formatted. Run 'cargo fmt --all' to fix.\n"
fi

# Gate 2: cargo clippy
echo "Gate 2/4: cargo clippy"
if ! cargo clippy --workspace -- -D warnings 2>&1; then
  ERRORS="${ERRORS}CLIPPY: Clippy warnings found. Fix all warnings before completing.\n"
fi

# Gate 3: cargo check (JSON error detection, tolerant of cargo exit code)
echo "Gate 3/4: cargo check"
CARGO_CHECK_OUT=$(cargo check --workspace --message-format=json 2>/dev/null || true)
if echo "$CARGO_CHECK_OUT" | jq -e 'select(.reason == "compiler-message" and .message.level == "error")' > /dev/null 2>&1; then
  ERRORS="${ERRORS}COMPILE: Compilation errors detected. Fix all errors before completing.\n"
fi

# Gate 4: cargo test
echo "Gate 4/4: cargo test"
if ! cargo test --workspace 2>&1; then
  ERRORS="${ERRORS}TESTS: Test failures detected. All tests must pass before completing.\n"
fi

# Gate 5: clean working tree
echo "Checking git status..."
UNCOMMITTED=$(git status --porcelain 2>/dev/null || echo "")
if [ -n "$UNCOMMITTED" ]; then
  ERRORS="${ERRORS}GIT: Uncommitted changes detected. Commit all changes before completing.\n${UNCOMMITTED}\n"
fi

if [ -n "$ERRORS" ]; then
  echo ""
  echo "=== QUALITY GATE FAILURES ==="
  echo -e "$ERRORS"
  echo "Task completion rejected. Fix the above issues and try again."
  exit 2
fi

echo "All quality gates passed."
exit 0
