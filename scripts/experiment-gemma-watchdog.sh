#!/bin/bash
# Gemma-4-31B-it experiment watchdog.
#
# Lifecycle:
#   1. Wait until vasp-02 is idle (no active inference requests)
#   2. Swap in Gemma via experiment-gemma-swap.sh start
#   3. Monitor health every 30 minutes for EXPERIMENT_HOURS
#   4. Swap back to Devstral+OmniCoder
#
# Usage:
#   nohup ./scripts/experiment-gemma-watchdog.sh > /tmp/gemma-experiment.log 2>&1 &
#
# Monitor:
#   tail -f /tmp/gemma-experiment.log
#   cat /tmp/gemma-experiment-status.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_HOURS="${EXPERIMENT_HOURS:-12}"
MONITOR_INTERVAL_SECS="${MONITOR_INTERVAL_SECS:-1800}"  # 30 minutes
IDLE_CHECK_INTERVAL=30  # seconds between idle checks
MAX_IDLE_WAIT=7200      # 2 hours max wait for idle before giving up
STATUS_FILE="/tmp/gemma-experiment-status.json"
LOG_FILE="/tmp/gemma-experiment.log"
VASP02="10.0.0.21"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

write_status() {
    local phase="$1" msg="$2"
    cat > "$STATUS_FILE" <<EOJSON
{
  "phase": "$phase",
  "message": "$msg",
  "timestamp": "$(date -Iseconds)",
  "experiment_hours": $EXPERIMENT_HOURS,
  "started_at": "${EXPERIMENT_START:-null}",
  "ends_at": "${EXPERIMENT_END:-null}",
  "checks_completed": ${CHECKS_DONE:-0},
  "checks_failed": ${CHECKS_FAILED:-0},
  "vasp02_healthy": ${VASP02_HEALTHY:-true}
}
EOJSON
}

check_vasp02_idle() {
    # Check if vasp-02's llama-server has active slots (processing requests)
    local metrics
    metrics=$(curl -s --connect-timeout 5 "http://${VASP02}:8081/metrics" 2>/dev/null || echo "")
    if [ -z "$metrics" ]; then
        return 1  # Can't reach — treat as busy
    fi

    # llama.cpp /metrics exposes slots_processing gauge
    local processing
    processing=$(echo "$metrics" | grep 'llamacpp_slots_processing{' | awk '{print $NF}' | head -1)
    if [ -z "$processing" ]; then
        # Fallback: check GPU utilization
        local gpu_util
        gpu_util=$(ssh -o ConnectTimeout=5 "root@${VASP02}" \
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "99")
        [ "${gpu_util%% *}" -lt 5 ]
        return $?
    fi

    # 0 slots processing = idle
    [ "${processing%%.*}" -eq 0 ] 2>/dev/null
}

check_gemma_health() {
    # Returns 0 if Gemma is responding on vasp-02:8081
    local health
    health=$(curl -s --connect-timeout 5 "http://${VASP02}:8081/health" 2>/dev/null || echo "")
    echo "$health" | grep -q "ok"
}

check_tz_routing() {
    # Verify TZ config has gemma variants in experiments
    grep -q "gemma_31b_worker" "$REPO_ROOT/config/tensorzero.toml" 2>/dev/null && \
    grep "candidate_variants.*worker" "$REPO_ROOT/config/tensorzero.toml" | grep -q "gemma_31b_worker"
}

count_gemma_inferences() {
    # Query TZ postgres for inference count on gemma variants since experiment start
    local count
    count=$(docker exec tensorzero-postgres-1 psql -U tensorzero -d tensorzero -t -c \
        "SELECT COUNT(*) FROM tensorzero.chat_inferences
         WHERE variant_name IN ('gemma_31b_worker', 'gemma_31b_fixer')
         AND created_at >= '${EXPERIMENT_START}'" 2>/dev/null | tr -d ' ')
    echo "${count:-0}"
}

# ════════════════════════════════════════════════════════════════════════
# Phase 1: Wait for vasp-02 idle
# ════════════════════════════════════════════════════════════════════════
log "=== Gemma Experiment Watchdog ==="
log "Experiment duration: ${EXPERIMENT_HOURS}h, monitor interval: $((MONITOR_INTERVAL_SECS/60))min"
write_status "waiting_for_idle" "Waiting for vasp-02 to finish active work"

waited=0
while ! check_vasp02_idle; do
    if [ $waited -ge $MAX_IDLE_WAIT ]; then
        log "ERROR: vasp-02 not idle after $((MAX_IDLE_WAIT/60))min. Aborting."
        write_status "aborted" "vasp-02 never became idle"
        exit 1
    fi
    if [ $((waited % 300)) -eq 0 ]; then
        log "Waiting for vasp-02 idle... (${waited}s / ${MAX_IDLE_WAIT}s max)"
    fi
    sleep "$IDLE_CHECK_INTERVAL"
    waited=$((waited + IDLE_CHECK_INTERVAL))
done

log "vasp-02 is idle after ${waited}s wait."

# ════════════════════════════════════════════════════════════════════════
# Phase 2: Swap in Gemma
# ════════════════════════════════════════════════════════════════════════
write_status "swapping_in" "Swapping Devstral+OmniCoder → Gemma on vasp-02"
log "Swapping in Gemma-4-31B-it..."

if ! bash "$REPO_ROOT/scripts/experiment-gemma-swap.sh" start; then
    log "ERROR: Swap failed!"
    write_status "swap_failed" "experiment-gemma-swap.sh start failed"
    exit 1
fi

# Verify Gemma is actually running
sleep 5
if ! check_gemma_health; then
    log "ERROR: Gemma not healthy after swap!"
    log "Rolling back..."
    bash "$REPO_ROOT/scripts/experiment-gemma-swap.sh" stop
    write_status "swap_failed" "Gemma not healthy after start, rolled back"
    exit 1
fi

EXPERIMENT_START="$(date -Iseconds)"
EXPERIMENT_END="$(date -Iseconds -d "+${EXPERIMENT_HOURS} hours")"
CHECKS_DONE=0
CHECKS_FAILED=0
VASP02_HEALTHY=true

log "Gemma experiment STARTED at $EXPERIMENT_START"
log "Scheduled end: $EXPERIMENT_END"
write_status "running" "Gemma experiment active"

# ════════════════════════════════════════════════════════════════════════
# Phase 3: Monitor loop
# ════════════════════════════════════════════════════════════════════════
end_epoch=$(date -d "+${EXPERIMENT_HOURS} hours" +%s)

while [ "$(date +%s)" -lt "$end_epoch" ]; do
    sleep "$MONITOR_INTERVAL_SECS"
    CHECKS_DONE=$((CHECKS_DONE + 1))

    remaining_h=$(( (end_epoch - $(date +%s)) / 3600 ))
    remaining_m=$(( ((end_epoch - $(date +%s)) % 3600) / 60 ))

    # Health check
    if check_gemma_health; then
        VASP02_HEALTHY=true
        inferences=$(count_gemma_inferences)
        log "CHECK #${CHECKS_DONE}: Gemma healthy | inferences=${inferences} | remaining=${remaining_h}h${remaining_m}m"
    else
        VASP02_HEALTHY=false
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        log "WARNING CHECK #${CHECKS_DONE}: Gemma NOT healthy! (failure ${CHECKS_FAILED})"

        # Try to recover if endpoint crashed
        if [ "$CHECKS_FAILED" -ge 3 ]; then
            log "ERROR: 3+ consecutive health failures. Aborting experiment."
            write_status "aborted_unhealthy" "Gemma failed ${CHECKS_FAILED} consecutive health checks"
            break
        fi
    fi

    # Verify TZ routing
    if ! check_tz_routing; then
        log "WARNING: TZ config doesn't show gemma in experiments! Config may have been reverted."
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    else
        # Reset failure counter on success
        if $VASP02_HEALTHY; then
            CHECKS_FAILED=0
        fi
    fi

    write_status "running" "Check #${CHECKS_DONE} | inferences=${inferences:-?} | remaining=${remaining_h}h${remaining_m}m"
done

# ════════════════════════════════════════════════════════════════════════
# Phase 4: Swap back
# ════════════════════════════════════════════════════════════════════════
log "Experiment period complete. Final inference count: $(count_gemma_inferences)"
write_status "swapping_out" "Restoring Devstral+OmniCoder"

log "Swapping back to Devstral+OmniCoder..."
if bash "$REPO_ROOT/scripts/experiment-gemma-swap.sh" stop; then
    log "Swap back successful."
    write_status "completed" "Experiment finished. $(count_gemma_inferences) inferences collected over ${EXPERIMENT_HOURS}h."
else
    log "ERROR: Swap back failed! Manual intervention needed."
    write_status "swap_back_failed" "experiment-gemma-swap.sh stop failed — vasp-02 may be in bad state"
    exit 1
fi

log "=== Experiment Complete ==="
log "Duration: ${EXPERIMENT_HOURS}h"
log "Health checks: ${CHECKS_DONE} (${CHECKS_FAILED} failures)"
log "Review results: TZ UI at http://localhost:4000 → filter variant=gemma_31b_worker"
