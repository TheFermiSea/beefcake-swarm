#!/usr/bin/env bash
# Phase 2 Python-backed single-issue runner. Replaces scripts/run-swarm.sh
# for the mini-SWE-agent path. Keeps the old script alongside until we're
# confident the new path is stable.
#
# Usage:
#   ./scripts/run-mini.sh --issue beefcake-abc123
#   ./scripts/run-mini.sh --issue manual-probe --objective 'Reply with OK'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/python/.venv"

# ── Core env (shared with legacy script) ──────────────────────────────────────
export BD_ACTOR="${BD_ACTOR:-swarm-$(hostname -s 2>/dev/null || echo worker)}"
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-$SCRIPT_DIR/bd-safe.sh}"

# Default cloud endpoint (CLIAPIProxy). Set SWARM_CLOUD_URL="" to use local.
if [[ -z "${SWARM_CLOUD_URL+x}" ]]; then
  export SWARM_CLOUD_URL="http://localhost:8317/v1"
fi
if [[ -n "${SWARM_CLOUD_URL:-}" ]]; then
  : "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
  export SWARM_CLOUD_API_KEY
fi
export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"

# CLIAPIProxy-routed models aren't in LiteLLM's pricing DB; ignore cost errors.
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

# ── Python env ────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Python venv not found at $VENV. Run:" >&2
  echo "  cd $REPO_ROOT/python && uv sync" >&2
  exit 1
fi

exec "$VENV/bin/python" "$REPO_ROOT/python/run.py" "$@"
