#!/usr/bin/env bash
# postmortem-review.sh — Cloud council reviews a failed dogfood run.
#
# Sends the last portion of the run log to Claude for diagnosis,
# updates the beads issue with findings, and resets it to open.
#
# Usage: postmortem-review.sh <issue-id> <log-file>
#
# Requires: SWARM_CLOUD_API_KEY, SWARM_CLOUD_URL (or defaults to CLIAPIProxy)
set -euo pipefail

ISSUE_ID="${1:?Usage: postmortem-review.sh <issue-id> <log-file>}"
LOG_FILE="${2:?Usage: postmortem-review.sh <issue-id> <log-file>}"

CLOUD_URL="${SWARM_CLOUD_URL:-http://localhost:8317/v1}"
CLOUD_KEY="${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY required}"
# Ordered fallback list — tried in sequence until one returns non-empty content.
# Primary model from env, then hardcoded fallbacks that are known-good.
POSTMORTEM_MODELS=(
  "${SWARM_CLOUD_MODEL:-claude-opus-4-6}"
  "claude-opus-4-6"
  "claude-sonnet-4-6"
  "gemini-3-flash-preview"
)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEADS_BIN="${SWARM_BEADS_BIN:-${SCRIPT_DIR}/bd-safe.sh}"

log() { echo "[postmortem $(date -Iseconds)] $*"; }

if [[ ! -f "$LOG_FILE" ]]; then
  log "Log file not found: $LOG_FILE — skipping postmortem"
  exit 0
fi

# Extract the last 200 lines of the run log (enough for error context,
# small enough to fit in a single API call).
LOG_TAIL="$(tail -200 "$LOG_FILE" 2>/dev/null)"

# Get the issue title and description
ISSUE_INFO="$("$BEADS_BIN" show "$ISSUE_ID" 2>/dev/null || echo "Issue: $ISSUE_ID")"

# Build the prompt for the cloud council
PROMPT_FILE="$(mktemp -t postmortem-prompt.XXXXXX)"
PAYLOAD_FILE="$(mktemp -t postmortem-payload.XXXXXX)"
RESPONSE_FILE="$(mktemp -t postmortem-response.XXXXXX)"
cleanup() {
  rm -f "$PROMPT_FILE" "$PAYLOAD_FILE" "$RESPONSE_FILE"
}
trap cleanup EXIT

cat >"$PROMPT_FILE" <<PROMPT_EOF
You are a senior engineering lead reviewing a failed automated coding attempt.

## Issue
$ISSUE_INFO

## Run Log (last 200 lines)
\`\`\`
$LOG_TAIL
\`\`\`

## Your Task

Analyze why this automated coding attempt failed. Provide:

1. **Root Cause** (1-2 sentences): What specifically went wrong?
2. **Pattern** (1 sentence): Is this a known failure pattern (e.g., edit_file mismatch, model hallucination, wrong file targeted, compilation cascade)?
3. **Fix Strategy** (2-3 bullet points): What should the next attempt do differently? Be specific — name exact files, functions, or approaches.
4. **Revised Description**: Rewrite the issue description to guide the next attempt. Include specific file paths, function signatures, and step-by-step instructions that avoid the failure pattern.

Respond in plain text (no markdown fences). Start with "ROOT CAUSE:" on the first line.
PROMPT_EOF

# Try each model in the cascade until one returns non-empty content.
DIAGNOSIS=""
USED_MODEL=""
# Deduplicate while preserving order (bash 4+)
declare -A _seen_models
DEDUPE_MODELS=()
for _m in "${POSTMORTEM_MODELS[@]}"; do
  if [[ -z "${_seen_models[$_m]+x}" ]]; then
    _seen_models[$_m]=1
    DEDUPE_MODELS+=("$_m")
  fi
done

for MODEL in "${DEDUPE_MODELS[@]}"; do
  log "Sending postmortem for $ISSUE_ID to $MODEL..."

  # Rebuild payload with the current model name.
  python3 - "$MODEL" "$PROMPT_FILE" "$PAYLOAD_FILE" <<'PY'
import json, pathlib, sys
model, prompt_path, payload_path = sys.argv[1:4]
prompt = pathlib.Path(prompt_path).read_text()
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2000,
    "temperature": 0.3,
}
pathlib.Path(payload_path).write_text(json.dumps(payload))
PY

  HTTP_STATUS="$(curl -sS --max-time 120 \
    -o "$RESPONSE_FILE" \
    -w "%{http_code}" \
    -H "x-api-key: $CLOUD_KEY" \
    -H "Content-Type: application/json" \
    "${CLOUD_URL%/}/chat/completions" \
    --data-binary "@$PAYLOAD_FILE")"

  if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
    log "Model $MODEL failed HTTP $HTTP_STATUS — trying next"
    continue
  fi

  DIAGNOSIS="$(python3 - "$RESPONSE_FILE" <<'PY'
import json, pathlib, sys
try:
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
except Exception:
    print("")
    raise SystemExit(0)
choices = payload.get("choices") or []
if not choices:
    print("")
    raise SystemExit(0)
message = choices[0].get("message") or {}
content = message.get("content")
if isinstance(content, str):
    print(content)
elif isinstance(content, list):
    parts = [item.get("text","") for item in content
             if isinstance(item, dict) and item.get("type") == "text" and item.get("text")]
    print("\n".join(parts))
else:
    print("")
PY
  )"

  if [[ -n "$DIAGNOSIS" ]]; then
    USED_MODEL="$MODEL"
    break
  fi

  log "Model $MODEL returned empty content — trying next"
  log "Raw response: $(python3 - "$RESPONSE_FILE" <<'PY'
import pathlib, sys
print(pathlib.Path(sys.argv[1]).read_text(errors='replace')[:300])
PY
  )"
done

if [[ -z "$DIAGNOSIS" ]]; then
  log "All postmortem models returned empty — skipping update"
  exit 0
fi
log "Received diagnosis via $USED_MODEL (${#DIAGNOSIS} chars)"

# Update the beads issue with the postmortem notes
NOTES="## Postmortem ($(date -u +%Y-%m-%dT%H:%M:%SZ))

$DIAGNOSIS"

"$BEADS_BIN" update "$ISSUE_ID" --notes="$NOTES" 2>/dev/null || \
  log "Failed to update issue notes (non-fatal)"

# Update the description with the revised version if present
REVISED_DESC="$(echo "$DIAGNOSIS" | sed -n '/Revised Description/,$ p' | tail -n +2)"
if [[ -n "$REVISED_DESC" && ${#REVISED_DESC} -gt 50 ]]; then
  "$BEADS_BIN" update "$ISSUE_ID" --description="$REVISED_DESC" 2>/dev/null || \
    log "Failed to update description (non-fatal)"
  log "Updated issue description with revised instructions"
fi

# Reset issue to open so it re-enters bd ready
"$BEADS_BIN" update "$ISSUE_ID" --status=open 2>/dev/null || \
  log "Failed to reset status (non-fatal)"

log "Postmortem complete for $ISSUE_ID — issue reopened with council guidance"
