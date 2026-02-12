#!/bin/bash
# HPC Cluster Watchdog (Deacon Equivalent)
# Continuous health monitoring for multi-agent system
#
# Inspired by Gastown's Deacon patrol pattern
# Monitors: Model endpoints, GPU health, SLURM jobs, Agent states

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/.beads/watchdog.log"
PATROL_INTERVAL="${PATROL_INTERVAL:-300}"  # 5 minutes default

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_endpoint() {
    local name="$1"
    local host="$2"
    local port="${3:-8080}"
    
    if curl -sf --max-time 5 "http://${host}:${port}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} ${name} (${host}:${port})"
        return 0
    else
        echo -e "${RED}✗${NC} ${name} (${host}:${port}) - UNREACHABLE"
        return 1
    fi
}

check_gpu_health() {
    local node="$1"
    local ip="$2"
    
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "root@${ip}" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} GPU on ${node}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} GPU check failed on ${node} (node may be unreachable)"
        return 1
    fi
}

check_slurm_jobs() {
    if ssh -o ConnectTimeout=5 -o BatchMode=yes root@10.0.0.5 \
        "squeue -p gpu_ai --noheader" 2>/dev/null | grep -q .; then
        local count=$(ssh root@10.0.0.5 "squeue -p gpu_ai --noheader | wc -l")
        echo -e "${GREEN}✓${NC} SLURM: ${count} jobs on gpu_ai partition"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} No jobs on gpu_ai partition"
        return 0  # Not necessarily an error
    fi
}

check_agent_beads() {
    cd "$PROJECT_ROOT"
    local agents=$(bd list --label gt:agent --json 2>/dev/null | jq -r '.[].id' 2>/dev/null || echo "")
    
    if [ -n "$agents" ]; then
        local count=$(echo "$agents" | wc -l | tr -d ' ')
        echo -e "${GREEN}✓${NC} Agent beads: ${count} registered"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} No agent beads found"
        return 0
    fi
}

create_alert_bead() {
    local title="$1"
    local description="$2"
    
    cd "$PROJECT_ROOT"
    bd create "$title" -t bug -p 0 --label watchdog --label alert -d "$description"
    log "ALERT: Created bead for: $title"
}

run_patrol() {
    log "=== Starting patrol cycle ==="
    local failures=0
    
    echo ""
    echo "=== HPC Cluster Health Check ==="
    echo ""
    
    # Check model endpoints
    echo "Model Endpoints:"
    check_endpoint "Strand-14B + Qwen3-Coder-Next (vasp-02)" "10.0.0.21" 8080 || ((failures++))
    check_endpoint "OR1-Behemoth-72B (vasp-01)" "10.0.0.20" 8081 || ((failures++))
    echo ""
    
    # Check GPU health (if reachable)
    echo "GPU Health:"
    check_gpu_health "vasp-01" "10.0.0.20" || true
    check_gpu_health "vasp-02" "10.0.0.21" || true
    check_gpu_health "vasp-03" "10.0.0.22" || true
    echo ""
    
    # Check SLURM
    echo "SLURM Status:"
    check_slurm_jobs || true
    echo ""
    
    # Check agent beads
    echo "Agent Beads:"
    check_agent_beads || true
    echo ""
    
    if [ $failures -gt 0 ]; then
        log "WARN: ${failures} endpoint(s) unreachable"
        echo -e "${YELLOW}⚠ ${failures} issue(s) detected${NC}"
    else
        log "OK: All endpoints healthy"
        echo -e "${GREEN}✓ All systems healthy${NC}"
    fi
    
    log "=== Patrol cycle complete ==="
}

daemon_mode() {
    log "Starting watchdog daemon (interval: ${PATROL_INTERVAL}s)"
    
    while true; do
        run_patrol
        log "Sleeping for ${PATROL_INTERVAL}s..."
        sleep "$PATROL_INTERVAL"
    done
}

# Main
case "${1:-once}" in
    once)
        run_patrol
        ;;
    daemon)
        daemon_mode
        ;;
    *)
        echo "Usage: $0 [once|daemon]"
        echo "  once   - Run single patrol cycle (default)"
        echo "  daemon - Run continuous patrol"
        exit 1
        ;;
esac
