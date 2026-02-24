#!/bin/bash
###############################################################################
# AI Inference Auto-Start Daemon
#
# TEMPORARY: This daemon auto-starts llama.cpp inference when VASP queue is idle.
# It should be REMOVED when MCP integration is complete (beefcake2-40om, y51x).
#
# Manages inference tiers:
#   - Fast+Coder (14B + Qwen3-Coder-Next router mode) on vasp-02 (:8080)
#   - Reasoning (72B Q4_K_M distributed) on vasp-01 + vasp-03 (:8081)
#   - Manager (Qwen3.5-397B MoE distributed) on all 3 nodes (:8081)
#   - Embedding (nomic-embed-code CPU-only) on vasp-02 (:8082)
#
# Tracked in: beads issue beefcake2-j9c3
#
# Usage:
#   ./ai-inference-daemon.sh              # Run in foreground
#   ./ai-inference-daemon.sh --once       # Single check (for cron)
#   systemctl start ai-inference-daemon   # If installed as service
###############################################################################

set -euo pipefail

# Configuration
CHECK_INTERVAL="${AI_DAEMON_INTERVAL:-60}"
SCRIPTS_PATH="${SLURM_SCRIPTS_PATH:-/cluster/shared/scripts/llama-cpp}"
ENDPOINTS_PATH="${SLURM_ENDPOINTS_PATH:-/cluster/shared/ai/endpoints}"
LOG_FILE="${AI_DAEMON_LOG:-/var/log/ai-inference-daemon.log}"
PID_FILE="/var/run/ai-inference-daemon.pid"

# VASP partitions to check
VASP_PARTITIONS="dev,normal,high"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

vasp_queue_active() {
    local count
    count=$(squeue -p "$VASP_PARTITIONS" -h -t RUNNING,PENDING 2>/dev/null | wc -l)
    [[ $count -gt 0 ]]
}

inference_job_exists() {
    local tier=$1
    local job_name
    case $tier in
        fast)      job_name="llama-14b" ;;
        reasoning) job_name="llama-72b" ;;
        manager)   job_name="llama-qwen35" ;;
        embed)     job_name="llama-embed" ;;
        *)         return 1 ;;
    esac
    squeue -n "$job_name" -h -t RUNNING,PENDING 2>/dev/null | grep -q .
}

inference_endpoint_healthy() {
    local tier=$1
    local pattern endpoint_file host port

    case $tier in
        fast)      pattern="*-14b.json" ;;
        reasoning) pattern="*-72b.json" ;;
        manager)   pattern="*-qwen35.json" ;;
        embed)     pattern="*-embed.json" ;;
        *)         return 1 ;;
    esac

    endpoint_file=$(ls -1t "$ENDPOINTS_PATH"/$pattern 2>/dev/null | head -1)
    [[ -z "$endpoint_file" ]] && return 1

    host=$(jq -r '.host // .head_node // empty' "$endpoint_file" 2>/dev/null)
    port=$(jq -r '.port // 8080' "$endpoint_file" 2>/dev/null)
    [[ -z "$host" ]] && return 1

    curl -sf --max-time 5 "http://${host}:${port}/health" >/dev/null 2>&1
}

submit_inference_job() {
    local tier=$1
    local script

    case $tier in
        fast)      script="run-14b.slurm" ;;
        reasoning) script="run-72b-distributed.slurm" ;;
        manager)   script="run-qwen35-distributed.slurm" ;;
        embed)     script="run-embedding.slurm" ;;
        *)         log "ERROR: Unknown tier: $tier"; return 1 ;;
    esac

    local script_path="${SCRIPTS_PATH}/${script}"
    if [[ ! -f "$script_path" ]]; then
        log "ERROR: Script not found: $script_path"
        return 1
    fi

    log "Submitting $tier inference job..."
    local job_id
    job_id=$(sbatch --parsable "$script_path" 2>&1)

    if [[ $job_id =~ ^[0-9]+$ ]]; then
        log "Submitted $tier job: $job_id"
        return 0
    else
        log "ERROR: Failed to submit job: $job_id"
        return 1
    fi
}

check_tier() {
    local tier=$1

    if inference_job_exists "$tier"; then
        if inference_endpoint_healthy "$tier"; then
            log "[$tier] healthy"
        else
            log "[$tier] job exists but endpoint not ready"
        fi
        return 0
    fi

    # No job running - start it
    log "[$tier] no job running - starting"
    submit_inference_job "$tier"
}

check_and_start_all() {
    # If VASP jobs are active, do nothing
    if vasp_queue_active; then
        log "VASP queue active, skipping AI inference check"
        return 0
    fi

    # Fast+coder tier (strand-14B + Qwen3-Coder-Next on vasp-02)
    check_tier "fast" || true
    # Reasoning tier (OR1-Behemoth 72B on vasp-01+03)
    check_tier "reasoning" || true
    # Manager tier (Qwen3.5-397B MoE on all 3 nodes)
    check_tier "manager" || true
    # Embedding tier (nomic-embed-code CPU-only on vasp-02)
    check_tier "embed" || true
}

# Single check mode
if [[ "${1:-}" == "--once" ]]; then
    check_and_start_all
    exit 0
fi

# Daemon mode
log "AI Inference Daemon starting (interval: ${CHECK_INTERVAL}s)"
log "Managing: fast+coder (vasp-02:8080) + reasoning (vasp-01,vasp-03:8081) + manager (all:8081, Qwen3.5-397B) + embed (vasp-02:8082, CPU-only)"
log "VASP partitions monitored: $VASP_PARTITIONS"

echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"; log "Daemon stopped"; exit 0' SIGTERM SIGINT

while true; do
    check_and_start_all || true
    sleep "$CHECK_INTERVAL"
done
