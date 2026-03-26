#!/usr/bin/env bash
# beads-sync.sh — Sync the active beads Dolt store to ai-proxy
#
# Usage:
#   scripts/beads-sync.sh          # push local → ai-proxy
#   scripts/beads-sync.sh pull     # pull ai-proxy → local
#
# In shared-server mode, the authoritative Dolt data lives under
# ~/.beads/shared-server/dolt on each machine, not under .beads/dolt in the repo.
# Stop the shared server cleanly before rsync so it reloads the new files on restart.
#
# This script exists because the swarm shells out to `bd`, and the shared-server
# Dolt state must be present on ai-proxy for the swarm to see local issues.

set -euo pipefail

REMOTE_HOST="${BEADS_SYNC_HOST:-brian@100.105.113.58}"
REMOTE_REPO_ROOT="${BEADS_SYNC_REPO_ROOT:-~/code/beefcake-swarm}"
LOCAL_REPO_ROOT="$(pwd)"

if grep -Eq '^dolt\.shared-server:\s*true$' .beads/config.yaml 2>/dev/null; then
    LOCAL_DIR="${BEADS_SYNC_LOCAL_DIR:-$HOME/.beads/shared-server/dolt/}"
    REMOTE_DIR="${BEADS_SYNC_DIR:-~/.beads/shared-server/dolt/}"
    MODE_LABEL="shared-server"
else
    LOCAL_DIR="${BEADS_SYNC_LOCAL_DIR:-.beads/dolt/}"
    REMOTE_DIR="${BEADS_SYNC_DIR:-$REMOTE_REPO_ROOT/.beads/dolt/}"
    MODE_LABEL="embedded"
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "Error: $LOCAL_DIR not found. Run from project root." >&2
    exit 1
fi

direction="${1:-push}"

case "$direction" in
    push)
        echo "Syncing local Dolt store ($MODE_LABEL) → $REMOTE_HOST:$REMOTE_DIR"
        if [[ "$MODE_LABEL" == "shared-server" ]]; then
            ssh "$REMOTE_HOST" "cd $REMOTE_REPO_ROOT && bd dolt stop" || true
        fi
        rsync -avz --delete "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR" | tail -5
        echo "Done. Run 'bd show <id>' on ai-proxy to verify."
        ;;
    pull)
        echo "Pulling Dolt store ($MODE_LABEL) from $REMOTE_HOST:$REMOTE_DIR → $LOCAL_DIR"
        if [[ "$MODE_LABEL" == "shared-server" ]]; then
            (
                cd "$LOCAL_REPO_ROOT"
                bd dolt stop
            ) || true
        fi
        rsync -avz --delete "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR" | tail -5
        echo "Done. Run 'bd show <id>' locally to verify."
        ;;
    *)
        echo "Usage: $0 [push|pull]" >&2
        exit 1
        ;;
esac
