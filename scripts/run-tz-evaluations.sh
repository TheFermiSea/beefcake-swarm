#!/usr/bin/env bash
# run-tz-evaluations.sh — Build fresh dataset + run TZ evaluations.
# Intended to run periodically (cron/systemd timer) to close the feedback loop.
set -euo pipefail

GATEWAY="http://localhost:3000"
PG_URL="postgres://tensorzero:tensorzero@localhost:5433/tensorzero"
CONFIG="/home/brian/code/beefcake-swarm/config/tensorzero.toml"
COMPOSE_DIR="/home/brian/code/beefcake-swarm/infrastructure/tensorzero"
DATASET_SUFFIX="$(date +%Y%m%d)"
LOG="/home/brian/tz-eval-${DATASET_SUFFIX}.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== TZ Evaluation Run ==="

# Helper: build dataset for a function from recent inferences (last 7 days)
build_dataset() {
  local func_name="$1" dataset_name="$2"
  log "Building dataset for ${func_name}..."
  docker exec tensorzero-postgres-1 psql -U tensorzero -d tensorzero -t -A -c "
  SELECT json_build_object(
    'datapoints', json_agg(json_build_object(
      'type', 'chat',
      'function_name', '${func_name}',
      'input', ci.input,
      'output', ci.output
    ))
  )
  FROM (
    SELECT ci.input, ci.output
    FROM chat_inferences ci
    WHERE ci.function_name = '${func_name}'
      AND ci.created_at > now() - interval '7 days'
    ORDER BY ci.created_at DESC
    LIMIT 20
  ) ci;
  " > /tmp/tz-eval-dataset-${func_name}.json

  local payload_size
  payload_size=$(wc -c < "/tmp/tz-eval-dataset-${func_name}.json")
  log "  ${func_name} dataset: ${payload_size} bytes"

  if [ "$payload_size" -lt 10 ]; then
    log "  No recent inferences for ${func_name} — skipping"
    return 1
  fi

  RESP=$(curl -s -X POST "$GATEWAY/v1/datasets/${dataset_name}/datapoints" \
    -H "Content-Type: application/json" \
    -d @"/tmp/tz-eval-dataset-${func_name}.json")
  log "  Dataset response: $(echo "$RESP" | head -c 200)"
  return 0
}

# Helper: run a single evaluation for a variant
run_eval() {
  local eval_name="$1" dataset_name="$2" variant="$3"
  log "  Running ${eval_name}: variant=${variant}"
  docker run --rm --network host \
    -v "$(dirname "$CONFIG"):/app/config:ro" \
    -e "SWARM_CLOUD_API_KEY=${SWARM_CLOUD_API_KEY}" \
    -e "TENSORZERO_POSTGRES_URL=${PG_URL}" \
    tensorzero/evaluations \
    --config-file /app/config/tensorzero.toml \
    --evaluation-name "$eval_name" \
    --dataset-name "$dataset_name" \
    --variant-name "$variant" \
    --concurrency 2 2>&1 | tee -a "$LOG"
}

# ── 1. worker_code_edit evaluations ──────────────────────────────────────────
WORKER_DATASET="worker_eval_${DATASET_SUFFIX}"
if build_dataset "worker_code_edit" "$WORKER_DATASET"; then
  for VARIANT in qwen35_27b devstral_24b sera_14b_worker; do
    log "Evaluating worker variant: $VARIANT"
    run_eval "worker_code_quality" "$WORKER_DATASET" "$VARIANT"
    run_eval "worker_behavior_quality" "$WORKER_DATASET" "$VARIANT"
  done
fi

# ── 2. code_fixing evaluations ───────────────────────────────────────────────
FIXER_DATASET="fixer_eval_${DATASET_SUFFIX}"
if build_dataset "code_fixing" "$FIXER_DATASET"; then
  for VARIANT in qwen35_fixer devstral_fixer sera_14b_fixer; do
    log "Evaluating fixer variant: $VARIANT"
    run_eval "code_fixing_quality" "$FIXER_DATASET" "$VARIANT"
  done
fi

# ── 3. cloud_manager_delegation evaluations ──────────────────────────────────
MANAGER_DATASET="manager_eval_${DATASET_SUFFIX}"
if build_dataset "cloud_manager_delegation" "$MANAGER_DATASET"; then
  run_eval "manager_delegation" "$MANAGER_DATASET" "opus_primary"
fi

log "=== Evaluation complete ==="
