#!/bin/bash
# bd-remote.sh — Proxy bd commands to ai-proxy where glibc is compatible.
#
# Rocky 8.8 on vasp nodes has glibc 2.28; the bd binary needs 2.32+.
# This script SSH-forwards all bd commands to ai-proxy (Ubuntu, glibc 2.35).
#
# Usage: Set SWARM_BEADS_BIN to this script's path on vasp nodes.
#   export SWARM_BEADS_BIN=/root/code/beefcake-swarm/scripts/bd-remote.sh
#
# The dogfood-loop.sh and run-swarm.sh already respect SWARM_BEADS_BIN.

set -euo pipefail

REMOTE_HOST="${BD_REMOTE_HOST:-brian@100.105.113.58}"
REMOTE_BD="${BD_REMOTE_BIN:-/home/brian/.local/bin/bd}"
REMOTE_REPO="${BD_REMOTE_REPO:-/home/brian/code/beefcake-swarm}"

# Forward BD_ACTOR if set (for per-node identity)
ACTOR_ENV=""
if [[ -n "${BD_ACTOR:-}" ]]; then
    ACTOR_ENV="BD_ACTOR=$BD_ACTOR"
fi

# Forward all arguments to remote bd
exec ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$REMOTE_HOST" \
    "cd $REMOTE_REPO && $ACTOR_ENV $REMOTE_BD $*"
