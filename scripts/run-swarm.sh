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

# Inference flows through TensorZero (localhost:3000) — see
# config/tensorzero.toml for variant A/B weights. TZ itself routes to local
# llama.cpp endpoints (free) by default or CLIAPIProxy (costs real money)
# when SWARM_USE_CLOUD=1.
export SWARM_TENSORZERO_URL="${SWARM_TENSORZERO_URL:-http://localhost:3000}"
if [[ "${SWARM_USE_CLOUD:-0}" == "1" ]]; then
  export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:-http://localhost:8317/v1}"
  : "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set for SWARM_USE_CLOUD=1}"
  export SWARM_CLOUD_API_KEY
  export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"
fi

# CLIAPIProxy-routed models aren't in LiteLLM's pricing DB; ignore cost errors.
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

# ── Python env ────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Python venv not found at $VENV. Run:" >&2
  echo "  cd $REPO_ROOT/python && uv sync" >&2
  exit 1
fi

exec "$VENV/bin/python" "$REPO_ROOT/python/run.py" "$@"
