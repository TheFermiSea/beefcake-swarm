#!/usr/bin/env bash
# deploy-lora.sh — Deploy a GGUF LoRA adapter to a running llama-server node.
#
# Copies the adapter file to the target node, restarts llama-server with
# --lora pointing to the adapter, and verifies the model responds.
# Updates SWARM_ADAPTER_ID so the swarm knows which adapter is active.
#
# Usage:
#   bash deploy-lora.sh --adapter /scratch/ai/adapters/agentic-v1.gguf --node vasp-03 --scale 1.0
#
# Options:
#   --adapter <path>    Path to GGUF LoRA adapter file (required)
#   --node <name>       Target node: vasp-01, vasp-02, vasp-03 (required)
#   --scale <float>     LoRA scaling factor (default: 1.0)
#   --no-restart        Copy adapter only, do not restart llama-server
#   --rollback          Restart llama-server without any LoRA adapter
#   --via-proxy         SSH to nodes via ai-proxy jump host (for remote access)
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

ADAPTER_PATH=""
NODE=""
SCALE="1.0"
NO_RESTART=false
ROLLBACK=false
VIA_PROXY=false
PROXY_HOST="brian@100.105.113.58"

# Node IP mapping
declare -A NODE_IPS
NODE_IPS[vasp-01]="10.0.0.20"
NODE_IPS[vasp-02]="10.0.0.21"
NODE_IPS[vasp-03]="10.0.0.22"

# ── CLI parsing ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter)     ADAPTER_PATH="$2"; shift 2 ;;
    --node)        NODE="$2"; shift 2 ;;
    --scale)       SCALE="$2"; shift 2 ;;
    --no-restart)  NO_RESTART=true; shift ;;
    --rollback)    ROLLBACK=true; shift ;;
    --via-proxy)   VIA_PROXY=true; shift ;;
    -h|--help)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$NODE" ]]; then
  echo "ERROR: --node is required (vasp-01, vasp-02, vasp-03)" >&2
  exit 1
fi

if [[ ! "${NODE_IPS[$NODE]+_}" ]]; then
  echo "ERROR: Unknown node '$NODE'. Valid: vasp-01, vasp-02, vasp-03" >&2
  exit 1
fi

if [[ "$ROLLBACK" == false && -z "$ADAPTER_PATH" ]]; then
  echo "ERROR: --adapter is required (or use --rollback)" >&2
  exit 1
fi

NODE_IP="${NODE_IPS[$NODE]}"
NODE_SSH="root@${NODE_IP}"
ADAPTER_DEST_DIR="/scratch/ai/adapters"

log() { echo "[deploy-lora] $(date +%H:%M:%S) $*"; }

# ── SSH helper ───────────────────────────────────────────────────────────────

run_on_node() {
  local cmd="$1"
  if $VIA_PROXY; then
    ssh "$PROXY_HOST" "ssh $NODE_SSH '$cmd'"
  else
    ssh "$NODE_SSH" "$cmd"
  fi
}

copy_to_node() {
  local src="$1" dest="$2"
  if $VIA_PROXY; then
    # scp through proxy jump host
    scp -o "ProxyJump=$PROXY_HOST" "$src" "${NODE_SSH}:${dest}"
  else
    scp "$src" "${NODE_SSH}:${dest}"
  fi
}

# ── Rollback mode ────────────────────────────────────────────────────────────

if $ROLLBACK; then
  log "Rolling back: restarting $NODE without LoRA adapter"

  # Find the current llama-server command line to extract model/port/flags
  CURRENT_CMD=$(run_on_node "ps -eo args= | grep 'llama-server' | grep -v grep | head -1" 2>/dev/null || true)
  if [[ -z "$CURRENT_CMD" ]]; then
    echo "ERROR: No llama-server running on $NODE — nothing to rollback" >&2
    exit 1
  fi

  # Strip --lora and --lora-scaled flags from the command
  CLEAN_CMD=$(echo "$CURRENT_CMD" | sed -E 's/--lora [^ ]+//g; s/--lora-scaled [^ ]+//g' | tr -s ' ')

  log "Restarting without LoRA: $CLEAN_CMD"
  run_on_node "pkill -f llama-server || true; sleep 3"
  run_on_node "HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache nohup $CLEAN_CMD > /tmp/llama-rollback.log 2>&1 &"

  log "Waiting for server to become healthy..."
  for i in $(seq 1 30); do
    if run_on_node "curl -sf http://localhost:8081/health" &>/dev/null; then
      log "Server healthy after ${i}s (no LoRA)"
      log "Clear SWARM_ADAPTER_ID in your env to complete rollback."
      exit 0
    fi
    sleep 2
  done

  echo "ERROR: Server did not become healthy after 60s" >&2
  log "Check logs: ssh $NODE_SSH 'tail -50 /tmp/llama-rollback.log'"
  exit 1
fi

# ── Validate adapter file ────────────────────────────────────────────────────

if [[ ! -f "$ADAPTER_PATH" ]]; then
  # Check if it exists on the node already (common for NFS-shared /scratch)
  if run_on_node "test -f '$ADAPTER_PATH'" 2>/dev/null; then
    log "Adapter found on $NODE at $ADAPTER_PATH (NFS/local)"
    REMOTE_ADAPTER="$ADAPTER_PATH"
  else
    echo "ERROR: Adapter file not found locally or on $NODE: $ADAPTER_PATH" >&2
    exit 1
  fi
else
  ADAPTER_FILENAME=$(basename "$ADAPTER_PATH")
  REMOTE_ADAPTER="${ADAPTER_DEST_DIR}/${ADAPTER_FILENAME}"

  # Check if already present on node (e.g., shared NFS)
  if run_on_node "test -f '$REMOTE_ADAPTER'" 2>/dev/null; then
    log "Adapter already present on $NODE: $REMOTE_ADAPTER"
  else
    ADAPTER_SIZE=$(du -h "$ADAPTER_PATH" | cut -f1)
    log "Copying adapter to $NODE ($ADAPTER_SIZE)..."
    run_on_node "mkdir -p $ADAPTER_DEST_DIR"
    copy_to_node "$ADAPTER_PATH" "$REMOTE_ADAPTER"
    log "Copy complete"
  fi
fi

if $NO_RESTART; then
  log "Adapter deployed to $NODE:$REMOTE_ADAPTER (--no-restart: skipping server restart)"
  exit 0
fi

# ── Discover running llama-server config ─────────────────────────────────────

log "Discovering current llama-server configuration on $NODE..."

CURRENT_CMD=$(run_on_node "ps -eo args= | grep 'llama-server' | grep -v grep | head -1" 2>/dev/null || true)
if [[ -z "$CURRENT_CMD" ]]; then
  echo "ERROR: No llama-server running on $NODE. Start inference first." >&2
  echo "  See: inference/start-inference-*.sh" >&2
  exit 1
fi

log "Current: $CURRENT_CMD"

# Strip any existing --lora/--lora-scaled flags before adding new ones
BASE_CMD=$(echo "$CURRENT_CMD" | sed -E 's/--lora [^ ]+//g; s/--lora-scaled [^ ]+//g' | tr -s ' ')

# Build new command with LoRA
NEW_CMD="${BASE_CMD} --lora ${REMOTE_ADAPTER} --lora-scaled ${SCALE}"

log "New:     $NEW_CMD"

# ── Restart llama-server ─────────────────────────────────────────────────────

log "Stopping current llama-server on $NODE..."
run_on_node "pkill -f llama-server || true; sleep 3"

# Verify it stopped
if run_on_node "pgrep -f llama-server" &>/dev/null; then
  log "WARNING: llama-server still running, sending SIGKILL..."
  run_on_node "pkill -9 -f llama-server || true; sleep 2"
fi

log "Starting llama-server with LoRA adapter..."
run_on_node "HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache nohup ${NEW_CMD} > /tmp/llama-lora.log 2>&1 &"

# ── Health check ─────────────────────────────────────────────────────────────

log "Waiting for server to become healthy..."
HEALTHY=false
for i in $(seq 1 45); do
  if run_on_node "curl -sf http://localhost:8081/health" &>/dev/null; then
    HEALTHY=true
    log "Server healthy after ${i}s"
    break
  fi
  sleep 2
done

if ! $HEALTHY; then
  echo "ERROR: Server did not become healthy after 90s" >&2
  log "Recent logs from $NODE:"
  run_on_node "tail -30 /tmp/llama-lora.log" 2>/dev/null || true
  exit 1
fi

# ── Verify inference works ───────────────────────────────────────────────────

log "Running inference smoke test..."
SMOKE_RESP=$(run_on_node "curl -sf http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\": \"test\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply OK\"}], \"max_tokens\": 8}'" 2>/dev/null || true)

if echo "$SMOKE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices')" &>/dev/null; then
  log "Smoke test passed"
else
  log "WARNING: Smoke test response unexpected (server may still be loading):"
  log "  $SMOKE_RESP"
fi

# ── Report ───────────────────────────────────────────────────────────────────

ADAPTER_NAME=$(basename "$REMOTE_ADAPTER" .gguf)

log ""
log "LoRA adapter deployed successfully:"
log "  Node:      $NODE ($NODE_IP)"
log "  Adapter:   $REMOTE_ADAPTER"
log "  Scale:     $SCALE"
log "  Adapter ID: $ADAPTER_NAME"
log ""
log "Set in your environment or run-swarm.sh:"
log "  export SWARM_ADAPTER_ID=$ADAPTER_NAME"
log ""
log "To rollback (remove LoRA):"
log "  bash scripts/deploy-lora.sh --node $NODE --rollback"
