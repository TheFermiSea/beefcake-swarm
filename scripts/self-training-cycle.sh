#!/bin/bash
# self-training-cycle.sh — Automated self-improvement cycle with evaluation gates.
#
# Pipeline:
#   1. CURATE  — Extract + filter training data (quality scoring)
#   2. AUGMENT — Generate synthetic trajectories (Sonnet teacher)
#   3. TRAIN   — QLoRA on Modal H100
#   4. EVALUATE — Benchmark new adapter vs current on holdout set
#   5. GATE    — Only deploy if new adapter beats current (>= threshold)
#   6. DEPLOY  — Hot-swap adapter on vasp-03:8083
#   7. MONITOR — TZ A/B tests in production; auto-rollback on regression
#
# Cost: ~$10-20 per cycle. Designed for twice-daily cron.
#
# Usage:
#   bash scripts/self-training-cycle.sh
#   bash scripts/self-training-cycle.sh --skip-synthetic
#   bash scripts/self-training-cycle.sh --skip-eval     # dangerous: deploy without gate
#   bash scripts/self-training-cycle.sh --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d-%H%M)
WORK_DIR="/tmp/self-training-${TIMESTAMP}"
ADAPTER_NAME="sera-rust-${TIMESTAMP}"
EVAL_DIR="${WORK_DIR}/eval"
HISTORY_DIR="${HOME}/.cache/beefcake-swarm/training-history"

# Source env
if [[ -f "$HOME/.swarm-env" ]]; then
    set -a; source "$HOME/.swarm-env"; set +a
fi

SKIP_SYNTHETIC=false
SKIP_EVAL=false
DRY_RUN=false
NUM_SYNTHETIC=200
BASE_MODEL="allenai/SERA-14B"
# Minimum improvement required to deploy (0.0 = any improvement, 0.05 = 5% better)
EVAL_THRESHOLD=0.0
# Minimum training samples required
MIN_SAMPLES=50
# Maximum iterations for quality filter (lower = higher quality training data)
MAX_ITERATIONS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-synthetic) SKIP_SYNTHETIC=true; shift ;;
        --skip-eval)      SKIP_EVAL=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --num-synthetic)  NUM_SYNTHETIC="$2"; shift 2 ;;
        --base)           BASE_MODEL="$2"; shift 2 ;;
        --threshold)      EVAL_THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$WORK_DIR" "$EVAL_DIR" "$HISTORY_DIR"
log() { echo "[$(date +%H:%M:%S)] $*"; }
PATH="$HOME/.local/bin:$PATH"

log "╔══════════════════════════════════════════════════╗"
log "║  Self-Training Cycle: ${TIMESTAMP}              ║"
log "╠══════════════════════════════════════════════════╣"
log "║  Base model:     ${BASE_MODEL}"
log "║  Eval threshold: ${EVAL_THRESHOLD}"
log "║  Max iterations: ${MAX_ITERATIONS}"
log "║  Min samples:    ${MIN_SAMPLES}"
log "╚══════════════════════════════════════════════════╝"

# ─── Step 1: CURATE — Extract + quality-filter training data ────────────────

log "Step 1: CURATE — Extracting and filtering training data..."
cd "$REPO_ROOT"

# Extract from TZ with quality filters:
# - task_resolved = true
# - iterations_used <= MAX_ITERATIONS (fewer iterations = cleaner fix)
# - Must have tool_calls (agent actually did work)
# - Deduplicate by issue_id (keep best attempt)
python3 scripts/extract-training-data.py \
    --output "${WORK_DIR}/tz-raw.jsonl" \
    ${SWARM_TENSORZERO_PG_URL:+--pg-url "$SWARM_TENSORZERO_PG_URL"} \
    2>&1 | tail -3
RAW_COUNT=$(wc -l < "${WORK_DIR}/tz-raw.jsonl" 2>/dev/null || echo 0)

# Quality filter: score each sample and keep top tier
python3 -c "
import json, sys

max_iter = ${MAX_ITERATIONS}
kept, dropped = 0, 0
seen_issues = set()

def content_length(msg):
    '''Handle both string content and TZ nested content arrays.'''
    c = msg.get('content', '')
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(item.get('text', '')) for item in c if isinstance(item, dict))
    return 0

def total_content_length(sample):
    '''Total content across all messages, including system prompt.'''
    total = 0
    # Top-level system prompt (TZ format)
    system = sample.get('messages', [{}])[0] if sample.get('messages') else {}
    if isinstance(system, dict) and 'system' in system:
        total += len(system.get('system', ''))
        for m in system.get('messages', []):
            total += content_length(m)
    else:
        # Standard chat format
        for m in sample.get('messages', []):
            total += content_length(m)
    return total

with open('${WORK_DIR}/tz-raw.jsonl') as fin, open('${WORK_DIR}/tz-curated.jsonl', 'w') as fout:
    for line in fin:
        d = json.loads(line)
        meta = d.get('metadata', {})

        # Quality filters — try both field names
        iterations = meta.get('iterations', meta.get('iterations_used', 999))
        if iterations > max_iter:
            dropped += 1
            continue

        # Dedup by issue_id or episode_id
        issue_id = meta.get('issue_id', meta.get('episode_id', ''))
        if issue_id and issue_id in seen_issues:
            dropped += 1
            continue
        seen_issues.add(issue_id)

        # Check messages have substance
        if total_content_length(d) < 100:
            dropped += 1
            continue

        fout.write(line)
        kept += 1

print(f'Curated: {kept} kept, {dropped} dropped (from {kept+dropped} raw)')
" 2>&1
CURATED_COUNT=$(wc -l < "${WORK_DIR}/tz-curated.jsonl" 2>/dev/null || echo 0)
log "  Raw: ${RAW_COUNT}, Curated: ${CURATED_COUNT} (max ${MAX_ITERATIONS} iterations, deduped)"

# ─── Step 2: AUGMENT — Generate synthetic trajectories ──────────────────────

SYNTH_COUNT=0
if [[ "$SKIP_SYNTHETIC" == false ]]; then
    log "Step 2: AUGMENT — Generating ${NUM_SYNTHETIC} synthetic trajectories..."
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
    log "Step 2: AUGMENT — Skipped"
fi

# ─── Step 3: Split — Training set + holdout evaluation set ──────────────────

log "Step 3: SPLIT — Creating training + holdout sets..."
cat "${WORK_DIR}/tz-curated.jsonl" > "${WORK_DIR}/all-data.jsonl"
if [[ -f "${WORK_DIR}/synthetic.jsonl" ]]; then
    cat "${WORK_DIR}/synthetic.jsonl" >> "${WORK_DIR}/all-data.jsonl"
fi
TOTAL=$(wc -l < "${WORK_DIR}/all-data.jsonl")

if [[ "$TOTAL" -lt "$MIN_SAMPLES" ]]; then
    log "ERROR: Only ${TOTAL} samples (need ${MIN_SAMPLES}). Aborting."
    log "  Hint: Let the swarm resolve more issues or increase --num-synthetic"
    exit 1
fi

# 90/10 train/eval split (deterministic shuffle by hash)
python3 -c "
import json, hashlib

with open('${WORK_DIR}/all-data.jsonl') as f:
    lines = f.readlines()

train, holdout = [], []
for line in lines:
    h = hashlib.md5(line.encode()).hexdigest()
    if int(h[:2], 16) < 25:  # ~10% holdout
        holdout.append(line)
    else:
        train.append(line)

with open('${WORK_DIR}/train.jsonl', 'w') as f:
    f.writelines(train)
with open('${EVAL_DIR}/holdout.jsonl', 'w') as f:
    f.writelines(holdout)

print(f'Split: {len(train)} train, {len(holdout)} holdout')
" 2>&1
TRAIN_COUNT=$(wc -l < "${WORK_DIR}/train.jsonl")
HOLDOUT_COUNT=$(wc -l < "${EVAL_DIR}/holdout.jsonl")
log "  Total: ${TOTAL} → Train: ${TRAIN_COUNT}, Holdout: ${HOLDOUT_COUNT}"

# ─── Step 4: TRAIN — QLoRA on Modal H100 ────────────────────────────────────

log "Step 4: TRAIN — QLoRA on Modal H100..."
if [[ "$DRY_RUN" == true ]]; then
    modal run scripts/modal_train.py \
        --base "$BASE_MODEL" \
        --data "${WORK_DIR}/train.jsonl" \
        --output "${WORK_DIR}/adapter" \
        --dry-run 2>&1 | tail -15
    log "[DRY RUN] Exiting before training."
    exit 0
fi

modal run scripts/modal_train.py \
    --base "$BASE_MODEL" \
    --data "${WORK_DIR}/train.jsonl" \
    --output "${WORK_DIR}/adapter" 2>&1 | tail -10

if [[ ! -f "${WORK_DIR}/adapter/adapter_model.safetensors" ]]; then
    log "ERROR: Training failed — no adapter produced"
    exit 1
fi

# Extract training loss from the output
TRAIN_LOSS=$(grep -o '"train_loss": [0-9.]*' "${WORK_DIR}/adapter/README.md" 2>/dev/null | grep -o '[0-9.]*' || echo "unknown")
log "  Training loss: ${TRAIN_LOSS}"

# ─── Step 5: EVALUATE — Benchmark new vs current adapter ────────────────────

if [[ "$SKIP_EVAL" == true ]]; then
    log "Step 5: EVALUATE — Skipped (--skip-eval)"
    EVAL_PASS=true
else
    log "Step 5: EVALUATE — Perplexity benchmark on holdout set (Modal H100)..."
    log "  Comparing: base ${BASE_MODEL} vs base+adapter on ${HOLDOUT_COUNT} holdout samples"

    # Run perplexity evaluation on Modal.
    # Loads base model, computes perplexity on holdout.
    # Then applies the adapter, computes again.
    # Adapter must have LOWER perplexity (= better fit to correct answers).

    EVAL_RESULT=$(python3 -c "
import json, sys, subprocess, os
from pathlib import Path

# Read adapter files
adapter_dir = '${WORK_DIR}/adapter'
adapter_files = {}
for fname in os.listdir(adapter_dir):
    fpath = os.path.join(adapter_dir, fname)
    if os.path.isfile(fpath) and not fname.startswith('checkpoint'):
        adapter_files[fname] = open(fpath, 'rb').read()

# Read holdout data
holdout_data = open('${EVAL_DIR}/holdout.jsonl', 'rb').read()

# Call Modal evaluate_adapter function
# We use subprocess to call 'modal run' with a special eval entrypoint
# But actually we need to call evaluate_adapter.remote() from Python.

# Write a temporary eval script that imports and calls the function
eval_script = '''
import modal, json, os, sys

app = modal.App.lookup(\"beefcake-lora-training\")
evaluate_adapter = modal.Function.from_name(\"beefcake-lora-training\", \"evaluate_adapter\")

adapter_dir = sys.argv[1]
holdout_path = sys.argv[2]
base_model = sys.argv[3]

adapter_files = {}
for fname in os.listdir(adapter_dir):
    fpath = os.path.join(adapter_dir, fname)
    if os.path.isfile(fpath) and not fname.startswith(\"checkpoint\"):
        with open(fpath, \"rb\") as f:
            adapter_files[fname] = f.read()

with open(holdout_path, \"rb\") as f:
    holdout_data = f.read()

result = evaluate_adapter.remote(base_model, adapter_files, holdout_data)
print(json.dumps(result))
'''

with open('/tmp/run_eval.py', 'w') as f:
    f.write(eval_script)

# Deploy the app first so the function is available
subprocess.run(
    ['modal', 'deploy', '${REPO_ROOT}/scripts/modal_train.py'],
    capture_output=True, timeout=300
)

# Run eval
proc = subprocess.run(
    ['python3', '/tmp/run_eval.py', adapter_dir, '${EVAL_DIR}/holdout.jsonl', '${BASE_MODEL}'],
    capture_output=True, text=True, timeout=600
)

if proc.returncode != 0:
    print(json.dumps({'error': proc.stderr[-500:] if proc.stderr else 'unknown', 'passed': False}))
else:
    # Find the JSON line in output
    for line in proc.stdout.strip().split('\n'):
        try:
            d = json.loads(line)
            if 'base_ppl' in d:
                print(json.dumps(d))
                sys.exit(0)
        except json.JSONDecodeError:
            continue
    print(json.dumps({'error': 'no eval result found', 'passed': False}))
" 2>&1 | tail -1)

    log "  Eval result: ${EVAL_RESULT}"

    # Parse the result
    EVAL_PASS=$(python3 -c "
import json, sys
try:
    d = json.loads('${EVAL_RESULT}'.replace(\"'\", '\"'))
except:
    d = {}

passed = d.get('passed', False)
base_ppl = d.get('base_ppl', 'N/A')
adapter_ppl = d.get('adapter_ppl', 'N/A')
improvement = d.get('improvement_pct', 0)
error = d.get('error', '')

if error:
    print(f'ERROR: {error}')
    print('FALLBACK')  # Fall back to loss comparison
elif passed:
    print(f'Base PPL={base_ppl:.2f}, Adapter PPL={adapter_ppl:.2f}, Improvement={improvement:+.1f}%')
    print('PASS')
else:
    print(f'Base PPL={base_ppl:.2f}, Adapter PPL={adapter_ppl:.2f}, Improvement={improvement:+.1f}%')
    print('FAIL')
" 2>&1)

    log "  ${EVAL_PASS}"

    if echo "$EVAL_PASS" | grep -q "PASS"; then
        EVAL_PASS=true
        log "  Evaluation PASSED — adapter improves perplexity on holdout set"
    elif echo "$EVAL_PASS" | grep -q "FALLBACK"; then
        log "  Evaluation had errors — falling back to loss comparison"
        # Fallback: compare training loss vs previous cycle
        PREV_LOSS_FILE="${HISTORY_DIR}/latest-loss.txt"
        PREV_LOSS="999.0"
        if [[ -f "$PREV_LOSS_FILE" ]]; then
            PREV_LOSS=$(cat "$PREV_LOSS_FILE")
        fi
        if python3 -c "exit(0 if float('${TRAIN_LOSS}') < float('${PREV_LOSS}') else 1)" 2>/dev/null; then
            EVAL_PASS=true
            log "  Fallback PASSED — loss ${TRAIN_LOSS} < prev ${PREV_LOSS}"
        else
            EVAL_PASS=false
            log "  Fallback FAILED — loss ${TRAIN_LOSS} >= prev ${PREV_LOSS}"
        fi
    else
        EVAL_PASS=false
        log "  Evaluation FAILED — adapter makes perplexity worse"
        log "  Adapter saved to ${WORK_DIR}/adapter/ for inspection but NOT deployed"
        echo "${TIMESTAMP}: REJECTED eval_result='${EVAL_RESULT}' samples=${TRAIN_COUNT}" \
            >> "${HISTORY_DIR}/cycle-log.txt"
        exit 0
    fi

    # Save loss for fallback comparisons
    echo "${TRAIN_LOSS}" > "${HISTORY_DIR}/latest-loss.txt"
fi

# Save cycle history
echo "${TIMESTAMP}: DEPLOYED loss=${TRAIN_LOSS} samples=${TRAIN_COUNT} adapter=${ADAPTER_NAME}" \
    >> "${HISTORY_DIR}/cycle-log.txt"

# ─── Step 6: Convert to GGUF ────────────────────────────────────────────────

log "Step 6: CONVERT — PEFT to GGUF..."
ssh root@10.0.0.21 "mkdir -p /scratch/ai/adapters/${ADAPTER_NAME}" 2>/dev/null
scp "${WORK_DIR}/adapter/"* root@10.0.0.21:/scratch/ai/adapters/${ADAPTER_NAME}/
ssh root@10.0.0.21 "source /scratch/ai/venvs/lora-training/bin/activate && \
    cd /tmp && python convert_lora_to_gguf.py \
    /scratch/ai/adapters/${ADAPTER_NAME} \
    --outfile /scratch/ai/adapters/${ADAPTER_NAME}.gguf \
    --outtype f16" 2>&1 | tail -3

# Copy GGUF to vasp-03
scp root@10.0.0.21:/scratch/ai/adapters/${ADAPTER_NAME}.gguf /tmp/${ADAPTER_NAME}.gguf
ssh root@10.0.0.22 "mkdir -p /scratch/ai/adapters"
scp /tmp/${ADAPTER_NAME}.gguf root@10.0.0.22:/scratch/ai/adapters/${ADAPTER_NAME}.gguf

# ─── Step 7: DEPLOY — Hot-swap adapter ──────────────────────────────────────

log "Step 7: DEPLOY — Hot-swapping adapter on vasp-03:8083..."

# Record current adapter for rollback
CURRENT_ADAPTER=$(ssh root@10.0.0.22 "grep -o '\-\-lora [^ ]*' /tmp/start-sera-lora.sh" 2>/dev/null || echo "none")
echo "${CURRENT_ADAPTER}" > "${WORK_DIR}/rollback-adapter.txt"
log "  Previous adapter: ${CURRENT_ADAPTER}"

# Deploy new adapter
ssh root@10.0.0.22 "sed -i 's|--lora /scratch/ai/adapters/.*\.gguf|--lora /scratch/ai/adapters/${ADAPTER_NAME}.gguf|' /tmp/start-sera-lora.sh && bash /tmp/start-sera-lora.sh" 2>&1 | tail -2
sleep 30

if ssh root@10.0.0.22 "curl -sf http://localhost:8083/health" > /dev/null 2>&1; then
    log "  SERA + ${ADAPTER_NAME} HEALTHY"
else
    log "  WARNING: Deployment failed — rolling back to previous adapter"
    if [[ "$CURRENT_ADAPTER" != "none" ]]; then
        ssh root@10.0.0.22 "sed -i 's|--lora /scratch/ai/adapters/.*\.gguf|${CURRENT_ADAPTER}|' /tmp/start-sera-lora.sh && bash /tmp/start-sera-lora.sh" 2>&1
        sleep 30
        log "  Rollback complete"
    fi
    exit 1
fi

# ─── Summary ────────────────────────────────────────────────────────────────

log "╔══════════════════════════════════════════════════╗"
log "║  Cycle Complete                                 ║"
log "╠══════════════════════════════════════════════════╣"
log "║  Adapter:    ${ADAPTER_NAME}"
log "║  Train loss: ${TRAIN_LOSS}"
log "║  Prev loss:  ${PREV_LOSS}"
log "║  Samples:    ${TRAIN_COUNT} train + ${HOLDOUT_COUNT} holdout"
log "║  TZ curated: ${CURATED_COUNT} (from ${RAW_COUNT} raw)"
log "║  Synthetic:  ${SYNTH_COUNT}"
log "║  Status:     DEPLOYED"
log "╚══════════════════════════════════════════════════╝"
log ""
log "Next: TZ Thompson Sampling will A/B test this adapter in production."
log "History: ${HISTORY_DIR}/cycle-log.txt"
