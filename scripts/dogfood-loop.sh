#!/usr/bin/env bash
# Phase 2 Python-backed continuous dogfood loop. Replaces the Rust invocation
# path in scripts/dogfood-loop.sh. Keeps the old script alongside until
# mini-SWE-agent has enough confirmed dogfood runs.
#
# Usage:
#   ./scripts/dogfood-mini.sh --discover --cooldown 120
#   ./scripts/dogfood-mini.sh --issue-list "id1 id2" --max-runs 5
#   ./scripts/dogfood-mini.sh --discover --parallel 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/python/.venv"

export BD_ACTOR="${BD_ACTOR:-swarm-$(hostname -s 2>/dev/null || echo worker)}"
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-$SCRIPT_DIR/bd-safe.sh}"

export SWARM_TENSORZERO_URL="${SWARM_TENSORZERO_URL:-http://localhost:3000}"
if [[ "${SWARM_USE_CLOUD:-0}" == "1" ]]; then
  export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:-http://localhost:8317/v1}"
  : "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set for SWARM_USE_CLOUD=1}"
  export SWARM_CLOUD_API_KEY
  export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"
fi
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Python venv not found at $VENV. Run:" >&2
  echo "  cd $REPO_ROOT/python && uv sync" >&2
  exit 1
fi

exec "$VENV/bin/python" "$REPO_ROOT/python/dogfood.py" "$@"
