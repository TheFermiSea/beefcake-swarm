#!/bin/bash
# TensorZero health check + auto-recovery for Docker-in-LXC environments.
# Runs as a systemd timer every 5 minutes.
#
# Fixes two problems:
# 1. Phantom Docker network endpoints (LXC veth cleanup failure)
# 2. TZ gateway crashes or OOM restarts

LOG="/var/log/tensorzero-health.log"
COMPOSE_DIR="/home/brian/code/beefcake-swarm/infrastructure/tensorzero"
export SWARM_CLOUD_API_KEY="rust-daq-proxy-key"

log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# Check gateway health
if curl -sf --max-time 5 http://localhost:3000/health > /dev/null 2>&1; then
    exit 0  # Healthy, nothing to do
fi

log "TZ gateway unhealthy — attempting recovery"

# Step 1: Try simple restart
cd "$COMPOSE_DIR" || exit 1
docker compose restart gateway 2>> "$LOG"
sleep 10

if curl -sf --max-time 5 http://localhost:3000/health > /dev/null 2>&1; then
    log "Recovery: simple restart succeeded"
    exit 0
fi

# Step 2: Full stack restart with network cleanup
log "Simple restart failed — doing full stack recovery"

docker compose down --remove-orphans 2>> "$LOG"

# Clean phantom endpoints from all tensorzero networks
for net in $(docker network ls --filter name=tensorzero -q 2>/dev/null); do
    for ep in $(docker network inspect "$net" 2>/dev/null | \
        python3 -c 'import json,sys; d=json.load(sys.stdin); [print(c["Name"]) for c in d[0].get("Containers",{}).values()]' 2>/dev/null); do
        log "Disconnecting phantom endpoint: $ep from network $net"
        docker network disconnect -f "$net" "$ep" 2>> "$LOG"
    done
    docker network rm "$net" 2>> "$LOG"
done

# Prune any dangling networks
docker network prune -f 2>> "$LOG"

# Restart
docker compose up -d 2>> "$LOG"
sleep 15

if curl -sf --max-time 5 http://localhost:3000/health > /dev/null 2>&1; then
    log "Recovery: full stack restart succeeded"
else
    log "Recovery FAILED — manual intervention needed"
fi
