#!/bin/bash
# Time-sliced Gemma-4-31B-it experiment on vasp-02.
#
# Usage:
#   ./scripts/experiment-gemma-swap.sh start   # Swap Devstral+OmniCoder → Gemma
#   ./scripts/experiment-gemma-swap.sh stop    # Swap Gemma → Devstral+OmniCoder
#   ./scripts/experiment-gemma-swap.sh status  # Check what's running on vasp-02
#
# What happens on "start":
#   1. Stops Devstral-24B (vasp-02:8081) and OmniCoder-9B (vasp-02:8083)
#   2. Starts Gemma-4-31B-it on vasp-02:8081
#   3. Updates TZ config: adds gemma variants to experiments, removes devstral+omnicoder
#   4. Restarts TZ gateway
#
# What happens on "stop":
#   1. Stops Gemma-4-31B-it (vasp-02:8081)
#   2. Restarts Devstral-24B (vasp-02:8081) and OmniCoder-9B (vasp-02:8083)
#   3. Restores TZ config: removes gemma variants, restores devstral+omnicoder
#   4. Restarts TZ gateway
set -euo pipefail

VASP02="root@10.0.0.21"
TZ_CONFIG="config/tensorzero.toml"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- Experiment candidate lines (for sed swaps) ---
# Normal operation:
#   worker_code_edit: ["omnicoder_9b", "qwen35_27b", "devstral_24b", "sera_14b_worker"]
#   code_fixing:      ["omnicoder_fixer", "qwen35_fixer", "devstral_fixer", "sera_14b_fixer"]
# Gemma experiment:
#   worker_code_edit: ["gemma_31b_worker", "qwen35_27b", "sera_14b_worker"]
#   code_fixing:      ["gemma_31b_fixer", "qwen35_fixer", "sera_14b_fixer"]

apply_experiment_profile() {
    # $1 = profile name (e.g., "gemma-experiment" or "normal")
    bash "$REPO_ROOT/scripts/tz-apply-experiment.sh" "$1"
}

status() {
    echo "=== vasp-02 GPU ==="
    ssh "$VASP02" "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader" 2>/dev/null || echo "(unreachable)"
    echo ""
    echo "=== vasp-02 llama-server processes ==="
    ssh "$VASP02" "ps -eo pid,rss,args | grep llama-server | grep -v grep" 2>/dev/null || echo "(none)"
    echo ""
    echo "=== TZ experiment config ==="
    grep 'candidate_variants.*worker\|candidate_variants.*fixer' "$TZ_CONFIG" | head -4
}

start_gemma() {
    echo ">>> Stopping Devstral + OmniCoder on vasp-02..."
    ssh "$VASP02" "
        # Kill all llama-server instances on ports 8081 and 8083
        pids=\$(ps -eo pid=,args= | awk '/llama-server.*--port (8081|8083)( |\$)/ { print \$1 }')
        if [ -n \"\$pids\" ]; then
            kill \$pids 2>/dev/null || true
            sleep 3
        fi
        echo 'Stopped.'
    "

    echo ">>> Starting Gemma-4-31B-it on vasp-02:8081..."
    ssh "$VASP02" "bash /tmp/start-inference-gemma4-31b.sh"

    echo ">>> Waiting for Gemma to load..."
    for i in $(seq 1 30); do
        if curl -s --connect-timeout 2 http://10.0.0.21:8081/health 2>/dev/null | grep -q ok; then
            echo "Gemma healthy after ${i}s"
            break
        fi
        sleep 2
    done

    echo ">>> Applying gemma-experiment profile..."
    apply_experiment_profile gemma-experiment

    echo ""
    echo "=== Gemma experiment ACTIVE ==="
    echo "Gemma-4-31B-it running on vasp-02:8081"
    echo "TZ routing: gemma_31b_worker + gemma_31b_fixer in experiments"
    echo "devstral_24b + omnicoder_9b removed from experiments"
    echo ""
    echo "To stop: ./scripts/experiment-gemma-swap.sh stop"
}

stop_gemma() {
    echo ">>> Stopping Gemma on vasp-02..."
    ssh "$VASP02" "
        pids=\$(ps -eo pid=,args= | awk '/llama-server.*--port 8081( |\$)/ { print \$1 }')
        if [ -n \"\$pids\" ]; then
            kill \$pids 2>/dev/null || true
            sleep 3
        fi
        echo 'Stopped.'
    "

    echo ">>> Restarting Devstral-24B on vasp-02:8081..."
    ssh "$VASP02" "bash /tmp/start-inference.sh"

    echo ">>> Restarting OmniCoder-9B on vasp-02:8083..."
    ssh "$VASP02" "bash /tmp/start-inference-omnicoder9b.sh"

    echo ">>> Waiting for models to load..."
    for i in $(seq 1 30); do
        devstral_ok=$(curl -s --connect-timeout 2 http://10.0.0.21:8081/health 2>/dev/null | grep -c ok || true)
        omnicoder_ok=$(curl -s --connect-timeout 2 http://10.0.0.21:8083/health 2>/dev/null | grep -c ok || true)
        if [ "$devstral_ok" -ge 1 ] && [ "$omnicoder_ok" -ge 1 ]; then
            echo "Both models healthy after ${i}s"
            break
        fi
        sleep 2
    done

    echo ">>> Applying normal profile..."
    apply_experiment_profile normal

    echo ""
    echo "=== Normal operation RESTORED ==="
    echo "Devstral-24B on vasp-02:8081, OmniCoder-9B on vasp-02:8083"
    echo "TZ routing: devstral_24b + omnicoder_9b restored to experiments"
}

case "${1:-}" in
    start)  start_gemma ;;
    stop)   stop_gemma ;;
    status) status ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo "  start  — Swap in Gemma-4-31B-it, update TZ experiments"
        echo "  stop   — Restore Devstral + OmniCoder, update TZ experiments"
        echo "  status — Show vasp-02 GPU, processes, and TZ config"
        exit 1
        ;;
esac
