#!/usr/bin/env bash
set -euo pipefail
export RUST_LOG="${RUST_LOG:-info}"
# Worker tier: HydraCoder on vasp-02
export SWARM_WORKER_URL="${SWARM_WORKER_URL:-http://vasp-02:8080/v1}"
export SWARM_WORKER_MODEL="${SWARM_WORKER_MODEL:-HydraCoder-Q6_K.gguf}"
# Local manager: Qwen3.5 distributed on vasp-01+vasp-03
export SWARM_LOCAL_MANAGER_URL="${SWARM_LOCAL_MANAGER_URL:-http://vasp-01:8081/v1}"
export SWARM_LOCAL_MANAGER_MODEL="${SWARM_LOCAL_MANAGER_MODEL:-Qwen3.5-397B-A17B-UD-Q4_K_XL.gguf}"
# Cloud manager via proxy
export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:-http://10.0.0.5:8317/v1}"
: "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
export SWARM_CLOUD_API_KEY
export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"
export SWARM_CLOUD_FALLBACK_MODEL="${SWARM_CLOUD_FALLBACK_MODEL:-claude-sonnet-4-5-20250929}"
if [[ "${SWARM_REQUIRE_ANTHROPIC_OWNERSHIP:-1}" == "1" ]]; then
  models_resp="$(mktemp)"
  if curl -sS -H "Authorization: Bearer $SWARM_CLOUD_API_KEY" \
    "${SWARM_CLOUD_URL%/}/models" > "$models_resp"; then
    model_owner="$(python3 - "$models_resp" "$SWARM_CLOUD_MODEL" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
model = sys.argv[2]
entry = next((m for m in doc.get("data", []) if m.get("id") == model), None)
print((entry or {}).get("owned_by", ""))
PY
)"
    if [[ -n "$model_owner" && "$model_owner" != "anthropic" ]]; then
      echo "Cloud model ${SWARM_CLOUD_MODEL} is owned_by=${model_owner}; falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
      export SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
    fi
  fi
  rm -f "$models_resp"
fi
if [[ "${SWARM_CLOUD_PREFLIGHT:-1}" == "1" ]]; then
  probe_req="$(mktemp)"
  probe_resp="${probe_req}.out"
  printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}\n' \
    "$SWARM_CLOUD_MODEL" > "$probe_req"
  probe_http="$(curl -sS -o "$probe_resp" -w "%{http_code}" \
    -H "Authorization: Bearer $SWARM_CLOUD_API_KEY" \
    -H "Content-Type: application/json" \
    "${SWARM_CLOUD_URL%/}/chat/completions" \
    -d @"$probe_req" || echo "000")"
  if [[ "$probe_http" != "200" ]] || grep -qiE 'auth_unavailable|quota_exhausted|resource_exhausted|exhausted your capacity|quota will reset' "$probe_resp"; then
    echo "Cloud model ${SWARM_CLOUD_MODEL} unavailable (http=${probe_http}); falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
    export SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
  fi
  rm -f "$probe_req" "$probe_resp"
fi
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-bd}"
exec cargo run -p swarm-agents "$@"
