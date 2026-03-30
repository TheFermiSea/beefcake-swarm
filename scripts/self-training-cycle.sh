#!/bin/bash
# self-training-cycle.sh — Automated self-improvement cycle.
# Extracts training data from TZ, generates synthetic trajectories,
# trains a LoRA adapter on Modal, converts to GGUF, and deploys.
#
# Designed to run weekly via cron or manually.
# Cost: ~$8-15 per cycle (Sonnet trajectories + Modal H100).
#
# Usage:
#   bash scripts/self-training-cycle.sh
#   bash scripts/self-training-cycle.sh --skip-synthetic  # use TZ data only
#   bash scripts/self-training-cycle.sh --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d-%H%M)
WORK_DIR="/tmp/self-training-${TIMESTAMP}"
ADAPTER_NAME="sera-rust-${TIMESTAMP}"

# Source env
if [[ -f "$HOME/.swarm-env" ]]; then
    set -a; source "$HOME/.swarm-env"; set +a
fi

SKIP_SYNTHETIC=false
DRY_RUN=false
NUM_SYNTHETIC=500
BASE_MODEL="allenai/SERA-14B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-synthetic) SKIP_SYNTHETIC=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --num-synthetic)  NUM_SYNTHETIC="$2"; shift 2 ;;
        --base)           BASE_MODEL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$WORK_DIR"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Self-Training Cycle: ${TIMESTAMP} ==="
log "  Base model: ${BASE_MODEL}"
log "  Work dir:   ${WORK_DIR}"

# Step 1: Extract training data from TZ Postgres
log "Step 1: Extracting verified completions from TensorZero..."
cd "$REPO_ROOT"
python3 scripts/extract-training-data.py \
    --output "${WORK_DIR}/tz-data.jsonl" 2>&1 | tail -3
TZ_COUNT=$(wc -l < "${WORK_DIR}/tz-data.jsonl" 2>/dev/null || echo 0)
log "  Extracted ${TZ_COUNT} samples from TZ"

# Step 2: Generate synthetic trajectories (optional)
if [[ "$SKIP_SYNTHETIC" == false ]]; then
    log "Step 2: Generating ${NUM_SYNTHETIC} synthetic trajectories with Sonnet..."
    TEACHER_API_KEY="${SWARM_CLOUD_API_KEY:-}" \
    TEACHER_API_URL="${SWARM_CLOUD_URL:-http://localhost:8317/v1}" \
    python3 scripts/generate-synthetic-trajectories.py \
        --model claude-sonnet-4-6 \
        --num "$NUM_SYNTHETIC" \
        --mode direct \
        --output "${WORK_DIR}/synthetic.jsonl" \
        --categories all 2>&1 | tail -5
    SYNTH_COUNT=$(wc -l < "${WORK_DIR}/synthetic.jsonl" 2>/dev/null || echo 0)
    log "  Generated ${SYNTH_COUNT} synthetic trajectories"
else
    SYNTH_COUNT=0
    log "Step 2: Skipped (--skip-synthetic)"
fi

# Step 3: Combine training data
log "Step 3: Combining training data..."
cat "${WORK_DIR}/tz-data.jsonl" > "${WORK_DIR}/combined.jsonl"
if [[ -f "${WORK_DIR}/synthetic.jsonl" ]]; then
    cat "${WORK_DIR}/synthetic.jsonl" >> "${WORK_DIR}/combined.jsonl"
fi
TOTAL=$(wc -l < "${WORK_DIR}/combined.jsonl")
log "  Total: ${TOTAL} samples (${TZ_COUNT} TZ + ${SYNTH_COUNT} synthetic)"

if [[ "$TOTAL" -lt 10 ]]; then
    log "ERROR: Too few samples (${TOTAL}). Need at least 10. Aborting."
    exit 1
fi

# Step 4: Train on Modal
log "Step 4: Training ${BASE_MODEL} LoRA on Modal H100..."
if [[ "$DRY_RUN" == true ]]; then
    modal run scripts/modal_train.py \
        --base "$BASE_MODEL" \
        --data "${WORK_DIR}/combined.jsonl" \
        --output "${WORK_DIR}/adapter" \
        --dry-run 2>&1 | tail -15
    log "[DRY RUN] Would train here. Exiting."
    exit 0
fi

modal run scripts/modal_train.py \
    --base "$BASE_MODEL" \
    --data "${WORK_DIR}/combined.jsonl" \
    --output "${WORK_DIR}/adapter" 2>&1 | tail -10

if [[ ! -f "${WORK_DIR}/adapter/adapter_model.safetensors" ]]; then
    log "ERROR: Training failed — no adapter produced"
    exit 1
fi
log "  Adapter saved to ${WORK_DIR}/adapter/"

# Step 5: Convert to GGUF
log "Step 5: Converting to GGUF..."
scp -r "${WORK_DIR}/adapter/" root@10.0.0.21:/scratch/ai/adapters/${ADAPTER_NAME}/
ssh root@10.0.0.21 "mkdir -p /scratch/ai/adapters/${ADAPTER_NAME}" 2>/dev/null
scp "${WORK_DIR}/adapter/"* root@10.0.0.21:/scratch/ai/adapters/${ADAPTER_NAME}/
ssh root@10.0.0.21 "source /scratch/ai/venvs/lora-training/bin/activate && \
    cd /tmp && python convert_lora_to_gguf.py \
    /scratch/ai/adapters/${ADAPTER_NAME} \
    --outfile /scratch/ai/adapters/${ADAPTER_NAME}.gguf \
    --outtype f16" 2>&1 | tail -3

# Copy GGUF to vasp-03 for deployment
scp root@10.0.0.21:/scratch/ai/adapters/${ADAPTER_NAME}.gguf /tmp/${ADAPTER_NAME}.gguf
ssh root@10.0.0.22 "mkdir -p /scratch/ai/adapters"
scp /tmp/${ADAPTER_NAME}.gguf root@10.0.0.22:/scratch/ai/adapters/${ADAPTER_NAME}.gguf
log "  GGUF adapter deployed to vasp-03"

# Step 6: Deploy (restart SERA with new LoRA)
log "Step 6: Deploying new adapter..."
ssh root@10.0.0.22 "sed -i 's|--lora /scratch/ai/adapters/.*\.gguf|--lora /scratch/ai/adapters/${ADAPTER_NAME}.gguf|' /tmp/start-sera-lora.sh && bash /tmp/start-sera-lora.sh" 2>&1 | tail -2
sleep 30
if ssh root@10.0.0.22 "curl -sf http://localhost:8083/health" > /dev/null 2>&1; then
    log "  SERA + ${ADAPTER_NAME} HEALTHY"
else
    log "  WARNING: Deployment may have failed — check vasp-03:8083"
fi

log "=== Cycle Complete ==="
log "  Adapter: ${ADAPTER_NAME}"
log "  Samples: ${TOTAL}"
log "  Work dir: ${WORK_DIR}"
log "  Next: TZ will A/B test the new adapter automatically"
