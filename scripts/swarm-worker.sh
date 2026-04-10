#!/usr/bin/env bash
# swarm-worker.sh — Autonomous swarm worker that runs on a vasp compute node.
#
# Auto-detects the local node identity, configures inference to use localhost,
# and runs the dogfood loop with local worktrees and full CPU resources.
#
# Designed to run as a systemd service (swarm-worker.service) or standalone.
# Each node picks issues independently via `bd ready --claim` — no central
# dispatcher needed. Beads atomic claiming prevents double-work.
#
# Environment (set by systemd or caller):
#   SWARM_CLOUD_URL          — CLIAPIProxy on ai-proxy (required)
#   SWARM_CLOUD_API_KEY      — API key for cloud proxy (required)
#   SWARM_TENSORZERO_URL     — TZ gateway on ai-proxy (optional)
#   SWARM_TENSORZERO_PG_URL  — TZ postgres on ai-proxy (optional)
#   RUST_LOG                 — Log level (default: info)

set -euo pipefail

# ── Auto-detect node identity ────────────────────────────────────────

HOSTNAME=$(hostname -s)
case "$HOSTNAME" in
    vasp-01) NODE_ID=vasp-01; FAST_MODEL="Qwen3.5-27B";    FAST_URL="http://localhost:8081/v1" ;;
    vasp-02) NODE_ID=vasp-02; FAST_MODEL="Devstral-Small-2-24B"; FAST_URL="http://localhost:8081/v1" ;;
    vasp-03) NODE_ID=vasp-03; FAST_MODEL="GLM-4.7-Flash";   FAST_URL="http://localhost:8081/v1" ;;
    *)
        echo "ERROR: Unknown hostname '$HOSTNAME'. Expected vasp-{01,02,03}." >&2
        exit 1
        ;;
esac

echo "[swarm-worker] Node: $NODE_ID ($HOSTNAME)"
echo "[swarm-worker] Local inference: $FAST_URL ($FAST_MODEL)"

# ── Configure environment ────────────────────────────────────────────

export PATH="$HOME/.cargo/bin:$PATH"
cd /root/code/beefcake-swarm

# Source cargo env if needed
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

# Pull latest code before starting
echo "[swarm-worker] Pulling latest code..."
git fetch origin 2>/dev/null && git reset --hard origin/main 2>/dev/null || {
    echo "[swarm-worker] WARN: git pull failed, using current state"
}
echo "[swarm-worker] Code: $(git log --oneline -1)"

# Rebuild if binary is older than source
BINARY="target/release/swarm-agents"
if [[ ! -f "$BINARY" ]] || [[ $(find crates/ coordination/ -name '*.rs' -newer "$BINARY" 2>/dev/null | head -1) ]]; then
    echo "[swarm-worker] Rebuilding swarm-agents..."
    export RUSTC_WRAPPER=$(command -v sccache 2>/dev/null || echo "")
    cargo build --release -p swarm-agents 2>&1 | tail -3
fi

# ── Local inference: all tiers point to localhost ─────────────────────
# Each vasp node runs ONE model on its GPU. The swarm uses that model
# for all local tiers (fast/coder/reasoning). Cloud handles management.

export SWARM_FAST_URL="$FAST_URL"
export SWARM_FAST_MODEL="$FAST_MODEL"
export SWARM_CODER_URL="$FAST_URL"
export SWARM_CODER_MODEL="$FAST_MODEL"
export SWARM_REASONING_URL="$FAST_URL"
export SWARM_REASONING_MODEL="$FAST_MODEL"

# Cloud via ai-proxy CLIAPIProxy
export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:?SWARM_CLOUD_URL required}"
export SWARM_CLOUD_API_KEY="${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY required}"
export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"
export SWARM_REQUIRE_ANTHROPIC_OWNERSHIP="${SWARM_REQUIRE_ANTHROPIC_OWNERSHIP:-0}"

# TensorZero on ai-proxy (optional)
export SWARM_TENSORZERO_URL="${SWARM_TENSORZERO_URL:-}"
export SWARM_TENSORZERO_PG_URL="${SWARM_TENSORZERO_PG_URL:-}"

# Worktrees: local disk (fast I/O, 36 cores for compilation)
export SWARM_WORKTREE_DIR="/tmp/swarm-wt"

# Beads: use remote proxy if local bd is incompatible (Rocky 8 glibc < 2.32)
if ! bd --version >/dev/null 2>&1; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -x "$SCRIPT_DIR/bd-remote.sh" ]]; then
        echo "[swarm-worker] Local bd incompatible — using bd-remote.sh proxy to ai-proxy"
        export SWARM_BEADS_BIN="$SCRIPT_DIR/bd-remote.sh"
    else
        echo "[swarm-worker] WARNING: bd not working and bd-remote.sh not found" >&2
    fi
fi

# Beads identity
export BD_ACTOR="worker-$NODE_ID"

# Parallel: 1 issue per node (the node IS the parallelism unit)
PARALLEL=1

# Use more retries since we have plenty of compute
export SWARM_MAX_RETRIES="${SWARM_MAX_RETRIES:-6}"

# Logging
export RUST_LOG="${RUST_LOG:-info,hyper=info,reqwest=info,h2=info,rustls=info,tower=info}"

# Sync beads before starting
bd dolt pull 2>/dev/null || echo "[swarm-worker] WARN: bd dolt pull failed"

# ── Preflight checks ────────────────────────────────────────────────

echo "[swarm-worker] Preflight checks..."

# Check local inference
if curl -sf -m 5 "$FAST_URL/../health" >/dev/null 2>&1 || curl -sf -m 5 "${FAST_URL%/v1}/health" >/dev/null 2>&1; then
    echo "[swarm-worker] Local inference: OK"
else
    echo "[swarm-worker] WARN: Local inference at $FAST_URL not responding"
    echo "[swarm-worker] Will run in cloud-only mode"
fi

# Check cloud proxy
if curl -sf -m 5 -H "x-api-key: $SWARM_CLOUD_API_KEY" "$SWARM_CLOUD_URL/../health" >/dev/null 2>&1; then
    echo "[swarm-worker] Cloud proxy: OK"
else
    echo "[swarm-worker] WARN: Cloud proxy at $SWARM_CLOUD_URL not responding"
fi

# Check Rust toolchain
cargo --version >/dev/null 2>&1 || { echo "ERROR: cargo not found"; exit 1; }
echo "[swarm-worker] Rust: $(rustc --version)"
echo "[swarm-worker] Cargo: $(cargo --version)"

echo "[swarm-worker] Starting dogfood loop (parallel=$PARALLEL, actor=$BD_ACTOR)"

# ── Run the loop ─────────────────────────────────────────────────────

# Stop sentinel: touch /tmp/{node}-dogfood-stop to gracefully stop
exec ./scripts/dogfood-loop.sh \
    --discover \
    --cooldown 60 \
    --parallel "$PARALLEL" \
    --max-issue-failures 3
