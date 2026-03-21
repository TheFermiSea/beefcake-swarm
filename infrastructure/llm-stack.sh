#!/usr/bin/env bash
# llm-stack.sh — Stable inference start/stop/status for compute nodes.
#
# Deploy to: /usr/local/sbin/llm-stack on vasp-01, vasp-02, vasp-03
#
# Usage:
#   llm-stack start   # Start current model
#   llm-stack stop    # Graceful stop (SIGTERM, wait up to 30s)
#   llm-stack status  # Show current model and health
#   llm-stack drain   # Stop and confirm stopped (for benchmark/maintenance)
#
# Environment:
#   LLM_START_SCRIPT   Path to model start script (default: /tmp/start-current-model.sh)
#   LLM_PORT           Health check port (default: 8081)
#
set -euo pipefail

PROCESS_NAME="llama-server-mmq"
START_SCRIPT="${LLM_START_SCRIPT:-/tmp/start-current-model.sh}"
PORT="${LLM_PORT:-8081}"
STOP_TIMEOUT=30

log() { echo "[llm-stack] $*"; }

do_stop() {
  if ! pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; then
    log "Not running"
    return 0
  fi

  log "Sending SIGTERM to $PROCESS_NAME..."
  pkill -TERM -f "$PROCESS_NAME" 2>/dev/null || true

  local waited=0
  while pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [[ $waited -ge $STOP_TIMEOUT ]]; then
      log "Timeout after ${STOP_TIMEOUT}s — force killing"
      pkill -9 -f "$PROCESS_NAME" 2>/dev/null || true
      sleep 2
      break
    fi
  done

  if pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; then
    log "ERROR: Failed to stop $PROCESS_NAME"
    return 1
  fi

  log "Stopped"
}

do_start() {
  if pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; then
    log "Already running (pid=$(pgrep -f "$PROCESS_NAME" | head -1))"
    return 0
  fi

  # Try known start scripts in order
  local script=""
  for candidate in "$START_SCRIPT" /tmp/start-qwen35-mmq.sh /tmp/start-current-model.sh; do
    if [[ -f "$candidate" ]]; then
      script="$candidate"
      break
    fi
  done

  if [[ -z "$script" ]]; then
    log "ERROR: No start script found. Tried: $START_SCRIPT, /tmp/start-qwen35-mmq.sh"
    return 1
  fi

  log "Starting with $script..."
  HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache bash "$script"

  # Wait for health (up to 60s)
  local waited=0
  while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [[ $waited -ge 60 ]]; then
      log "WARNING: Not healthy after 60s — check logs"
      return 1
    fi
  done

  log "Started and healthy on port $PORT"
}

do_status() {
  if pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; then
    local pid model
    pid=$(pgrep -f "$PROCESS_NAME" | head -1)
    model=$(ps aux | grep "$PROCESS_NAME" | grep -v grep | sed 's/.*--model //' | sed 's/ --.*//' | head -1)
    log "Running (pid=$pid)"
    log "Model: ${model:-unknown}"

    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      log "Health: OK (port $PORT)"
    else
      log "Health: NOT RESPONDING (port $PORT)"
    fi
  else
    log "Not running"
  fi
}

do_drain() {
  do_stop
  # Double-check with pgrep
  if pgrep -f "$PROCESS_NAME" >/dev/null 2>&1; then
    log "ERROR: Drain failed — process still running"
    return 1
  fi
  log "Drained — GPU is free"
}

case "${1:-status}" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  drain)  do_drain ;;
  *)
    echo "Usage: llm-stack {start|stop|status|drain}"
    exit 1
    ;;
esac
