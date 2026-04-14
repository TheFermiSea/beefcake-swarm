#!/usr/bin/env bash
# Ensure TensorZero gateway is running and healthy.
# Handles the Docker host-network stale-endpoint bug by using a standalone
# container name (tz-gateway) instead of docker compose's tensorzero-gateway-1.
#
# Usage: ./scripts/ensure-tensorzero.sh
# Called by dogfood-loop.sh at startup.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TZ_HEALTH_URL="http://localhost:3000/health"
TZ_CONTAINER="tz-gateway"
TZ_PG_CONTAINER="tensorzero-postgres-1"

# Check if already healthy
if curl -s --connect-timeout 3 "$TZ_HEALTH_URL" | grep -q '"gateway":"ok"' 2>/dev/null; then
    echo "[ensure-tz] Gateway healthy"
    exit 0
fi

echo "[ensure-tz] Gateway not healthy — starting..."

# Ensure postgres is running (via compose — it doesn't have the host-network bug)
if ! docker ps --format '{{.Names}}' | grep -q "$TZ_PG_CONTAINER"; then
    echo "[ensure-tz] Starting postgres..."
    cd "$REPO_ROOT/infrastructure/tensorzero"
    SWARM_CLOUD_API_KEY="${SWARM_CLOUD_API_KEY:-rust-daq-proxy-key}" \
        docker compose up -d postgres 2>&1 | sed 's/^/  /'
    # Wait for postgres healthy
    for i in $(seq 1 30); do
        if docker ps --filter "name=$TZ_PG_CONTAINER" --format '{{.Status}}' | grep -q healthy; then
            break
        fi
        sleep 1
    done
fi

# Run migrations (idempotent)
cd "$REPO_ROOT/infrastructure/tensorzero"
SWARM_CLOUD_API_KEY="${SWARM_CLOUD_API_KEY:-rust-daq-proxy-key}" \
    docker compose run --rm gateway-run-postgres-migrations 2>/dev/null || true

# Remove any existing gateway container (stale or crashed)
docker rm -f "$TZ_CONTAINER" 2>/dev/null || true
# Also clean up compose-named containers that may have stale endpoints
docker rm -f tensorzero-gateway-1 2>/dev/null || true

# Start gateway with a standalone name to avoid the host-network stale-endpoint bug.
# Using `docker run` instead of `docker compose up` because compose's
# network_mode:host creates endpoints that survive container removal and
# block recreation without a Docker daemon restart.
docker run -d \
    --name "$TZ_CONTAINER" \
    --network host \
    -v "$REPO_ROOT/config:/app/config:ro" \
    -e SWARM_CLOUD_API_KEY="${SWARM_CLOUD_API_KEY:-rust-daq-proxy-key}" \
    -e TENSORZERO_POSTGRES_URL="postgres://tensorzero:tensorzero@localhost:5433/tensorzero" \
    -e TENSORZERO_AUTOPILOT_API_KEY="${TENSORZERO_AUTOPILOT_API_KEY:-}" \
    --restart unless-stopped \
    tensorzero/gateway --config-file /app/config/tensorzero.toml \
    > /dev/null 2>&1

# Wait for healthy
for i in $(seq 1 15); do
    if curl -s --connect-timeout 2 "$TZ_HEALTH_URL" | grep -q '"gateway":"ok"' 2>/dev/null; then
        echo "[ensure-tz] Gateway started and healthy"
        exit 0
    fi
    sleep 1
done

echo "[ensure-tz] WARNING: Gateway started but health check timed out"
exit 1
