#!/usr/bin/env bash
# enrich-cognition.sh — Periodically enrich the Cognition Base from NotebookLM.
#
# Queries NotebookLM for insights about recent error patterns and
# stores the results as cognition items for future retrieval.
#
# Usage:
#   ./scripts/enrich-cognition.sh
#   ./scripts/enrich-cognition.sh --repo-root /path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    *)           echo "Unknown arg: $1"; exit 1 ;;
  esac
done

COGNITION_DIR="$REPO_ROOT/.swarm/cognition"
NLM_BIN="${SWARM_NLM_BIN:-nlm}"

log() { echo "[enrich] $(date +%H:%M:%S) $*"; }

mkdir -p "$COGNITION_DIR"

# Notebook IDs from notebook_registry.toml
PROJECT_BRAIN_ID="9ecc5027-d983-4461-bbf7-fbb389bfcf03"
DEBUGGING_KB_ID="8266d75a-0e25-4ea8-9785-106b127c50c6"

# Helper: query a notebook and extract the answer text.
# Returns empty string on failure (never crashes the script).
query_notebook() {
  local notebook_id="$1" query="$2"
  $NLM_BIN query notebook "$notebook_id" "$query" 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('value', {}).get('answer', ''))
except Exception:
    print('')
" 2>/dev/null || echo ""
}

# Helper: store a cognition item as JSONL.
store_item() {
  local item_id="$1" domain="$2" content="$3"
  python3 -c "
import json, sys, datetime
item = {
    'id': sys.argv[1],
    'content': sys.stdin.read(),
    'source': 'Notebook',
    'domain': sys.argv[2],
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'embedding': []
}
print(json.dumps(item))
" "$item_id" "$domain" <<< "$content" >> "$COGNITION_DIR/items.jsonl" 2>/dev/null
}

DATE_TAG=$(date +%Y%m%d)

# --- Query 1: Error pattern insights from Project Brain ---
log "Querying Project Brain for error pattern insights..."
RESULT=$(query_notebook "$PROJECT_BRAIN_ID" \
  "What are the most common error patterns in this codebase and what are the proven fix strategies for each?")

if [[ -n "$RESULT" && ${#RESULT} -gt 50 ]]; then
  log "Got ${#RESULT} chars of insights. Storing as cognition item."
  store_item "nlm-error-patterns-${DATE_TAG}" "error-patterns" "$RESULT"
  log "Stored error-patterns item."
else
  log "No useful response from Project Brain. Skipping."
fi

# --- Query 2: Harness configuration insights from Debugging KB ---
log "Querying Debugging KB for harness configuration insights..."
RESULT2=$(query_notebook "$DEBUGGING_KB_ID" \
  "What harness parameters work best for different error types? What write deadlines and tool restrictions are optimal?")

if [[ -n "$RESULT2" && ${#RESULT2} -gt 50 ]]; then
  log "Got ${#RESULT2} chars. Storing as cognition item."
  store_item "nlm-harness-config-${DATE_TAG}" "harness-config" "$RESULT2"
  log "Stored harness-config item."
else
  log "No useful response from Debugging KB. Skipping."
fi

log "Enrichment complete."
