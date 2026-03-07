#!/usr/bin/env bash
# beads-sync.sh — Sync local beads dolt DB to ai-proxy
#
# Usage:
#   scripts/beads-sync.sh          # push local → ai-proxy
#   scripts/beads-sync.sh pull     # pull ai-proxy → local
#
# The dolt server on the target machine must be stopped before rsync
# overwrites the database files. bd's idle-monitor auto-stops the server
# after inactivity, so this usually works. If it doesn't, the script
# kills the idle-monitor first.
#
# This script exists because bd and bdh use separate datastores:
#   - bd: reads/writes local .beads/dolt/<database>/ (dolt SQL)
#   - bdh: reads/writes BeadHub server (PostgreSQL)
# The swarm binary (BeadsBridge) shells out to bd, so issues must
# exist in the local dolt DB on ai-proxy for the swarm to find them.

set -euo pipefail

REMOTE_HOST="${BEADS_SYNC_HOST:-brian@100.105.113.58}"
REMOTE_DIR="${BEADS_SYNC_DIR:-~/code/beefcake-swarm/.beads/dolt/}"
LOCAL_DIR=".beads/dolt/"

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "Error: $LOCAL_DIR not found. Run from project root." >&2
    exit 1
fi

direction="${1:-push}"

case "$direction" in
    push)
        echo "Syncing local dolt DB → $REMOTE_HOST"
        # Stop dolt server on remote so it picks up new files on next start
        ssh "$REMOTE_HOST" 'pkill -f "bd dolt idle-monitor" 2>/dev/null; pkill -f "dolt sql-server" 2>/dev/null; sleep 1' || true
        rsync -avz --delete "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR" | tail -5
        echo "Done. Run 'bd show <id>' on ai-proxy to verify."
        ;;
    pull)
        echo "Pulling dolt DB from $REMOTE_HOST → local"
        # Stop local dolt server
        pkill -f "bd dolt idle-monitor" 2>/dev/null || true
        pkill -f "dolt sql-server" 2>/dev/null || true
        sleep 1
        rsync -avz --delete "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR" | tail -5
        echo "Done. Run 'bd show <id>' locally to verify."
        ;;
    *)
        echo "Usage: $0 [push|pull]" >&2
        exit 1
        ;;
esac
