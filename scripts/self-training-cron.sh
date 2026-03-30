#!/bin/bash
# Self-training cron wrapper — runs from ai-proxy.
#
# Data-gated: skips training if fewer than MIN_NEW_EPISODES new successful
# episodes have accumulated since the last training run. No point burning
# GPU time retraining on the same data.
#
# Override base model via BASE_MODEL env var in crontab.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:$PATH"
source ~/.swarm-env 2>/dev/null
source ~/.venvs/swarm-scripts/bin/activate 2>/dev/null
cd ~/code/beefcake-swarm

# Pull latest code (non-destructive)
git pull --ff-only 2>/dev/null

mkdir -p ~/logs
LOG_FILE=~/logs/self-training-$(date +%Y%m%d).log
HISTORY_DIR="${HOME}/.cache/beefcake-swarm/training-history"
mkdir -p "$HISTORY_DIR"

# --- Data gate: check if enough new data has accumulated ---
MIN_NEW_EPISODES="${MIN_NEW_EPISODES:-20}"
LAST_TRAIN_FILE="${HISTORY_DIR}/last-training-timestamp.txt"
PG_URL="${SWARM_TENSORZERO_PG_URL:-postgres://tensorzero:tensorzero@localhost:5433/tensorzero}"

LAST_TS="1970-01-01T00:00:00Z"
if [[ -f "$LAST_TRAIN_FILE" ]]; then
    LAST_TS=$(cat "$LAST_TRAIN_FILE")
fi

# Count new successful episodes since last training
NEW_EPISODES=$(python3 -c "
import psycopg2, sys
try:
    conn = psycopg2.connect('${PG_URL}')
    cur = conn.cursor()
    cur.execute('''
        SELECT COUNT(DISTINCT target_id)
        FROM tensorzero.boolean_metric_feedback
        WHERE metric_name = 'task_resolved'
          AND value = true
          AND created_at > %s::timestamptz
    ''', ('${LAST_TS}',))
    print(cur.fetchone()[0])
    conn.close()
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(0)
" 2>>"$LOG_FILE")

echo "[$(date +%H:%M:%S)] Data gate: ${NEW_EPISODES} new episodes since ${LAST_TS} (threshold: ${MIN_NEW_EPISODES})" >> "$LOG_FILE"

if [[ "$NEW_EPISODES" -lt "$MIN_NEW_EPISODES" ]]; then
    echo "[$(date +%H:%M:%S)] Skipping training — only ${NEW_EPISODES} new episodes (need ${MIN_NEW_EPISODES})" >> "$LOG_FILE"
    exit 0
fi

echo "[$(date +%H:%M:%S)] Data gate passed — ${NEW_EPISODES} new episodes, proceeding with training" >> "$LOG_FILE"

# Record training start time BEFORE training (so next check counts from now)
date -u +%Y-%m-%dT%H:%M:%SZ > "$LAST_TRAIN_FILE"

# Use BASE_MODEL from env if set, otherwise default to SERA-14B
EXTRA_ARGS=""
if [[ -n "${BASE_MODEL:-}" ]]; then
    EXTRA_ARGS="--base $BASE_MODEL"
fi

exec bash scripts/self-training-cycle.sh --num-synthetic 200 $EXTRA_ARGS \
  >> "$LOG_FILE" 2>&1
