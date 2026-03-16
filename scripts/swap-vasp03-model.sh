#!/usr/bin/env bash
# swap-vasp03-model.sh — Swap the model running on vasp-03 (Scout/Fast tier).
#
# Usage:
#   ./scripts/swap-vasp03-model.sh opus-distilled   # Switch to Opus-Distilled 27B
#   ./scripts/swap-vasp03-model.sh original          # Switch back to original 27B-Distilled
#   ./scripts/swap-vasp03-model.sh status             # Show current model
#
# Saves rollback info to /scratch/ai/rollback-vasp03.txt on vasp-03.
#
set -euo pipefail

PROXY_HOST="brian@100.105.113.58"
VASP03="root@10.0.0.22"
LLAMA_BIN="/usr/local/bin/llama-server-mmq"
MODELS_DIR="/scratch/ai/models"

# Model definitions
declare -A MODEL_FILES
MODEL_FILES[opus-distilled]="Qwen3.5-27B-Opus-Distilled.Q4_K_M.gguf"
MODEL_FILES[original]="Qwen3.5-27B-Distilled-Q4_K_M.gguf"

declare -A MODEL_CTX
MODEL_CTX[opus-distilled]=65536
MODEL_CTX[original]=65536

ACTION="${1:-status}"

log() { echo "[swap] $*"; }

run_on_vasp03() {
    ssh "$PROXY_HOST" "ssh $VASP03 '$1'"
}

case "$ACTION" in
    status)
        log "Current model on vasp-03:"
        run_on_vasp03 "ps aux | grep llama-server | grep -v grep | sed 's/.*--model /  model: /;s/ --.*//' || echo '  not running'"
        run_on_vasp03 "curl -s http://localhost:8081/health 2>/dev/null && echo ' (healthy)' || echo '  (not responding)'"
        ;;

    opus-distilled|original)
        MODEL_FILE="${MODEL_FILES[$ACTION]}"
        CTX="${MODEL_CTX[$ACTION]}"

        log "Swapping to: $ACTION ($MODEL_FILE, ctx=$CTX)"

        # Save current state for rollback
        CURRENT_CMD=$(run_on_vasp03 "ps aux | grep llama-server | grep -v grep | sed 's/.*llama-server/llama-server/'" 2>/dev/null || echo "")
        if [[ -n "$CURRENT_CMD" ]]; then
            log "Saving rollback: $CURRENT_CMD"
        fi

        # Stop current
        log "Stopping current model..."
        run_on_vasp03 "pkill -f llama-server-mmq || true; sleep 3"

        # Start new
        log "Starting $ACTION..."
        run_on_vasp03 "HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache nohup $LLAMA_BIN --model $MODELS_DIR/$MODEL_FILE --host 0.0.0.0 --port 8081 --ctx-size $CTX --n-gpu-layers 99 -fa on > /tmp/llama-${ACTION}.log 2>&1 & echo \"PID: \$!\"; sleep 15; curl -s http://localhost:8081/health || echo 'NOT HEALTHY'"

        log "Swap complete. Verify: curl http://vasp-03:8081/health"
        ;;

    *)
        echo "Usage: $0 {opus-distilled|original|status}"
        exit 1
        ;;
esac
