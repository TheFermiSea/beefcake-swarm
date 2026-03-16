#!/usr/bin/env bash
# deploy-dogfood.sh — One-command deploy + restart of the dogfood loop on ai-proxy.
#
# Usage:
#   ./scripts/deploy-dogfood.sh                          # Default issue list
#   ./scripts/deploy-dogfood.sh --issues "id1 id2 id3"   # Custom issues
#   ./scripts/deploy-dogfood.sh --stop                    # Stop only, no restart
#
# What it does:
#   1. Kills any existing dogfood-loop processes on ai-proxy
#   2. Pulls latest code from git
#   3. Rebuilds the swarm-agents binary
#   4. Starts a single dogfood-loop instance with proper env vars
#
# Requirements:
#   - SSH access to brian@100.105.113.58 (ai-proxy)
#   - SWARM_CLOUD_API_KEY set in ai-proxy's ~/.bashrc
#
set -euo pipefail

PROXY_HOST="brian@100.105.113.58"
REPO_DIR="~/code/beefcake-swarm"
DEFAULT_ISSUES="beefcake-s9pz beefcake-rt4g beefcake-8do8 beefcake-jljc beefcake-lf7s beefcake-uv3z beefcake-4ony"
COOLDOWN=120
STOP_ONLY=false
ISSUES=""

# CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --issues)   ISSUES="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --stop)     STOP_ONLY=true; shift ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ISSUES="${ISSUES:-$DEFAULT_ISSUES}"

log() { echo "[deploy] $*"; }

# ── Step 1: Stop existing processes ──
log "Stopping existing dogfood processes on ai-proxy..."
ssh "$PROXY_HOST" "
  pkill -9 -f dogfood-loop 2>/dev/null || true
  sleep 1
  # Also kill any orphaned swarm-agents binaries
  pkill -9 -f 'target.*swarm-agents' 2>/dev/null || true
  rm -f /tmp/dogfood-loop.lock
  # Clean stale worktrees
  rm -rf /tmp/beefcake-wt/beefcake-* 2>/dev/null || true
  cd $REPO_DIR && git worktree prune 2>/dev/null || true
  echo 'Processes killed, worktrees cleaned'
" 2>/dev/null || true

if $STOP_ONLY; then
  log "Stop-only mode. Done."
  exit 0
fi

# ── Step 2: Pull + rebuild ──
log "Pulling latest code and rebuilding..."
ssh "$PROXY_HOST" "
  cd $REPO_DIR && \
  git pull --ff-only && \
  cargo build -p swarm-agents 2>&1 | tail -3
"

# ── Step 3: Start dogfood loop ──
log "Starting dogfood loop with issues: $ISSUES"
log "Cooldown: ${COOLDOWN}s"

# Use bash -l to source ~/.bashrc (which has SWARM_CLOUD_API_KEY)
ssh "$PROXY_HOST" "
  cd $REPO_DIR && \
  nohup bash -l -c '
    export SWARM_CLOUD_URL=http://localhost:8317/v1
    export SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0
    export RUST_LOG=debug,hyper=info,reqwest=info,h2=info,rustls=info,tower=info
    ./scripts/dogfood-loop.sh --issue-list \"$ISSUES\" --cooldown $COOLDOWN
  ' > ~/dogfood-\$(date +%Y%m%d-%H%M).log 2>&1 &
  sleep 3
  echo '=== LOCKFILE ==='
  cat /tmp/dogfood-loop.lock 2>/dev/null || echo 'no lockfile'
  echo '=== PROCESSES ==='
  ps aux | grep -E 'dogfood-loop|target.*swarm' | grep -v grep || echo 'none'
  echo '=== LOG START ==='
  tail -5 ~/dogfood-*.log 2>/dev/null | tail -5
"

log "Dogfood loop deployed. Monitor with:"
log "  ssh $PROXY_HOST 'tail -f ~/dogfood-*.log'"
