#!/usr/bin/env bash
set -euo pipefail
export RUST_LOG="${RUST_LOG:-info}"
export SWARM_USE_BDH="${SWARM_USE_BDH:-1}"
export SWARM_BDH_BIN="${SWARM_BDH_BIN:-bdh}"
# Scout/Fast tier: Qwen3.5-27B-Distilled on vasp-03 (VRAM-resident, 192K context)
export SWARM_FAST_URL="${SWARM_FAST_URL:-http://vasp-03:8081/v1}"
export SWARM_FAST_MODEL="${SWARM_FAST_MODEL:-Qwen3.5-27B-Distilled}"
# Coder/Integrator tier: Qwen3.5-122B-A10B MoE on vasp-01 (expert-offload, 65K context)
export SWARM_CODER_URL="${SWARM_CODER_URL:-http://vasp-01:8081/v1}"
export SWARM_CODER_MODEL="${SWARM_CODER_MODEL:-Qwen3.5-122B-A10B}"
# Reasoning tier: Qwen3.5-122B-A10B MoE on vasp-02 (independent instance, expert-offload, 65K context)
export SWARM_REASONING_URL="${SWARM_REASONING_URL:-http://vasp-02:8081/v1}"
export SWARM_REASONING_MODEL="${SWARM_REASONING_MODEL:-Qwen3.5-122B-A10B}"
# Cloud manager via CLIAPIProxy
# Set SWARM_CLOUD_URL="" (empty) to run in worker-first mode (local models only).
# Default to localhost when running on ai-proxy (where the proxy lives).
# Use http://10.0.0.5:8317/v1 via SWARM_CLOUD_URL override when running from compute nodes.
if [[ -z "${SWARM_CLOUD_URL+x}" ]]; then
  # Not set at all — default to localhost proxy
  export SWARM_CLOUD_URL="http://localhost:8317/v1"
elif [[ -z "$SWARM_CLOUD_URL" ]]; then
  # Explicitly set to empty — worker-first mode, unset so config.rs sees None
  unset SWARM_CLOUD_URL
fi

if [[ -n "${SWARM_CLOUD_URL:-}" ]]; then
  # Cloud mode: require API key and run preflight checks
  : "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
  export SWARM_CLOUD_API_KEY
  # Default to antigravity-hosted models (routed via CLIAPIProxy)
  export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-opus-4-6}"
  export SWARM_CLOUD_FALLBACK_MODEL="${SWARM_CLOUD_FALLBACK_MODEL:-claude-sonnet-4-5-20250929}"
  # CLIAPIProxy v6.8+ uses x-api-key header (not Authorization: Bearer)
  _PROXY_AUTH=(-H "x-api-key: $SWARM_CLOUD_API_KEY")
  if [[ "${SWARM_REQUIRE_ANTHROPIC_OWNERSHIP:-1}" == "1" ]]; then
    models_resp="$(mktemp)"
    if curl -sS "${_PROXY_AUTH[@]}" \
      "${SWARM_CLOUD_URL%/}/models" > "$models_resp"; then
      model_owner="$(python3 - "$models_resp" "$SWARM_CLOUD_MODEL" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
model = sys.argv[2]
entry = next((m for m in doc.get("data", []) if m.get("id") == model), None)
print((entry or {}).get("owned_by", ""))
PY
)"
      if [[ -n "$model_owner" && "$model_owner" != "anthropic" && "$model_owner" != "antigravity" ]]; then
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
      "${_PROXY_AUTH[@]}" \
      -H "Content-Type: application/json" \
      "${SWARM_CLOUD_URL%/}/chat/completions" \
      -d @"$probe_req" || echo "000")"
    if [[ "$probe_http" != "200" ]] || grep -qiE 'auth_unavailable|quota_exhausted|resource_exhausted|exhausted your capacity|quota will reset' "$probe_resp"; then
      echo "Cloud model ${SWARM_CLOUD_MODEL} unavailable (http=${probe_http}); falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
      export SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
    fi
    rm -f "$probe_req" "$probe_resp"
  fi
else
  echo "Worker-first mode: SWARM_CLOUD_URL not set, using local models only"
fi
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-bdh}"

# ── sccache: shared C/C++ compilation cache ──
# Eliminates redundant proc-macro and native dep builds across worktrees.
# Install: cargo install sccache
if command -v sccache &>/dev/null; then
    export RUSTC_WRAPPER=sccache
    export SCCACHE_DIR="${SCCACHE_DIR:-/tmp/beefcake-sccache}"
    mkdir -p "$SCCACHE_DIR"
fi

# ── Shared target directory ──
# Multiple worktrees share one target dir to avoid redundant dep builds.
# Cargo handles concurrent access with its own locking.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/beefcake-shared-target}"
mkdir -p "$CARGO_TARGET_DIR"

exec cargo run -p swarm-agents -- "$@"
