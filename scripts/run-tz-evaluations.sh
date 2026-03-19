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

# Build fresh worker dataset from recent inferences (last 24h, max 10)
log "Building worker dataset..."
docker exec tensorzero-postgres-1 psql -U tensorzero -d tensorzero -t -A -c "
SELECT json_build_object(
  'datapoints', json_agg(json_build_object(
    'type', 'chat',
    'function_name', 'worker_code_edit',
    'input', d.input,
    'output', d.output
  ))
)
FROM (
  SELECT d.input, d.output
  FROM chat_inference_data d
  JOIN chat_inferences i ON d.id = i.id AND d.created_at = i.created_at
  WHERE i.function_name = 'worker_code_edit'
    AND i.created_at > now() - interval '7 days'
  ORDER BY i.created_at DESC
  LIMIT 10
) d;
" > /tmp/tz-eval-dataset.json

PAYLOAD_SIZE=$(wc -c < /tmp/tz-eval-dataset.json)
log "Dataset payload: ${PAYLOAD_SIZE} bytes"

if [ "$PAYLOAD_SIZE" -lt 10 ]; then
  log "No recent inferences — skipping evaluation"
  exit 0
fi

DATASET_NAME="worker_eval_${DATASET_SUFFIX}"
RESP=$(curl -s -X POST "$GATEWAY/v1/datasets/${DATASET_NAME}/datapoints" \
  -H "Content-Type: application/json" \
  -d @/tmp/tz-eval-dataset.json)
log "Dataset response: $(echo "$RESP" | head -c 200)"

# Run evaluations for both variants
for VARIANT in qwen_coder qwen_reasoning; do
  log "Running worker_code_quality eval: variant=$VARIANT"
  docker run --rm --network host \
    -v "$(dirname "$CONFIG"):/app/config:ro" \
    -e "SWARM_CLOUD_API_KEY=${SWARM_CLOUD_API_KEY}" \
    -e "TENSORZERO_POSTGRES_URL=${PG_URL}" \
    tensorzero/evaluations \
    --config-file /app/config/tensorzero.toml \
    --evaluation-name worker_code_quality \
    --dataset-name "$DATASET_NAME" \
    --variant-name "$VARIANT" \
    --concurrency 2 2>&1 | tee -a "$LOG"
done

log "=== Evaluation complete ==="
