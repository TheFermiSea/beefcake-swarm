#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUST_LOG="${RUST_LOG:-info}"
# Native beads: swarm uses `bd` directly (BeadHub removed).
# BD_ACTOR identifies this orchestrator instance for `bd mail` messaging.
# Defaults to hostname-based identity; override for multi-instance setups.
export BD_ACTOR="${BD_ACTOR:-swarm-$(hostname -s 2>/dev/null || echo worker)}"
# Scout/Fast tier: GLM-4.7-Flash on vasp-03 (30B/3B MoE, SOTA tool-calling, ~50 tok/s, 32K context)
export SWARM_FAST_URL="${SWARM_FAST_URL:-http://vasp-03:8081/v1}"
export SWARM_FAST_MODEL="${SWARM_FAST_MODEL:-OmniCoder-9B}"
# Coder tier: Qwen3.5-27B on vasp-01 (dense, GPU-resident, ~27 tok/s, 32K context)
export SWARM_CODER_URL="${SWARM_CODER_URL:-http://vasp-01:8081/v1}"
export SWARM_CODER_MODEL="${SWARM_CODER_MODEL:-Qwen3.5-27B}"
# Reasoning tier: Devstral-Small-2-24B on vasp-02 (agentic coding, ~30 tok/s, 32K context)
export SWARM_REASONING_URL="${SWARM_REASONING_URL:-http://vasp-02:8081/v1}"
export SWARM_REASONING_MODEL="${SWARM_REASONING_MODEL:-Qwen3.5-27B}"
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

probe_cloud_model() {
  local model="$1"
  local probe_req probe_resp
  probe_req="$(mktemp)"
  probe_resp="${probe_req}.out"
  printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}\n' \
    "$model" > "$probe_req"
  PROBE_HTTP="$(curl -sS -o "$probe_resp" -w "%{http_code}" \
    "${_PROXY_AUTH[@]}" \
    -H "Content-Type: application/json" \
    "${SWARM_CLOUD_URL%/}/chat/completions" \
    -d @"$probe_req" || echo "000")"
  PROBE_BODY="$(cat "$probe_resp" 2>/dev/null || true)"
  rm -f "$probe_req" "$probe_resp"

  if [[ "$PROBE_HTTP" != "200" ]] || grep -qiE 'auth_unavailable|quota_exhausted|resource_exhausted|exhausted your capacity|quota will reset' <<<"$PROBE_BODY"; then
    return 1
  fi
  return 0
}

if [[ -n "${SWARM_CLOUD_URL:-}" ]]; then
  # Cloud mode: require API key and run preflight checks
  : "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
  export SWARM_CLOUD_API_KEY
  # Default primary cloud model routed via CLIAPIProxy; override SWARM_CLOUD_MODEL to switch.
  # gpt-5.4 routes through OpenAI/ChatGPT-Plus credentials (independent of Anthropic quota).
  export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-gpt-5.4}"
  export SWARM_CLOUD_FALLBACK_MODEL="${SWARM_CLOUD_FALLBACK_MODEL:-gpt-5.2-codex}"
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
      if [[ -n "$model_owner" && "$model_owner" != "anthropic" && "$model_owner" != "antigravity" && "$model_owner" != "openai" ]]; then
        echo "Cloud model ${SWARM_CLOUD_MODEL} is owned_by=${model_owner}; falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
        export SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
      fi
    fi
    rm -f "$models_resp"
  fi
  if [[ "${SWARM_CLOUD_PREFLIGHT:-1}" == "1" ]]; then
    if ! probe_cloud_model "$SWARM_CLOUD_MODEL"; then
      echo "Cloud model ${SWARM_CLOUD_MODEL} unavailable (http=${PROBE_HTTP}); probing fallback ${SWARM_CLOUD_FALLBACK_MODEL}"
      export SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
      if ! probe_cloud_model "$SWARM_CLOUD_MODEL"; then
        echo "Cloud fallback ${SWARM_CLOUD_MODEL} also unavailable (http=${PROBE_HTTP}); switching to worker-first local mode"
        unset SWARM_CLOUD_URL
        export SWARM_WORKER_FIRST_ENABLED="${SWARM_WORKER_FIRST_ENABLED:-1}"
      fi
    fi
  fi
else
  echo "Worker-first mode: SWARM_CLOUD_URL not set, using local models only"
fi
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-$SCRIPT_DIR/bd-safe.sh}"

# ── TensorZero feedback loop ──
# Enable the inference → feedback → optimize loop by telling the orchestrator
# where the TZ gateway and Postgres live. Without this, inferences are recorded
# but evaluations/episodes never populate.
# Use ${VAR-default} (no colon) so explicitly setting SWARM_TENSORZERO_URL=""
# disables TZ routing (config.rs filters empty strings).
export SWARM_TENSORZERO_URL="${SWARM_TENSORZERO_URL-http://localhost:3000}"
export SWARM_TENSORZERO_PG_URL="${SWARM_TENSORZERO_PG_URL-postgres://tensorzero:tensorzero@localhost:5433/tensorzero}"
# Auto-detect repo ID for TZ project isolation. Prevents cross-project
# feedback contamination when the swarm works on multiple codebases.
# Defaults to the basename of --repo-root or the current directory.
if [[ -z "${SWARM_REPO_ID-}" ]]; then
  if [[ -n "${_REPO_ROOT:-}" ]]; then
    export SWARM_REPO_ID="$(basename "$_REPO_ROOT")"
  else
    export SWARM_REPO_ID="$(basename "$(pwd)")"
  fi
fi

# ── Beads OpenTelemetry ──
# Export beads operational metrics (issue counts, storage ops) to OTLP collector.
# Complements TensorZero inference metrics.
if [[ -n "${SWARM_OTEL_ENDPOINT:-}" ]]; then
    "$SWARM_BEADS_BIN" config set telemetry.endpoint "$SWARM_OTEL_ENDPOINT" 2>/dev/null || true
    "$SWARM_BEADS_BIN" config set telemetry.enabled true 2>/dev/null || true
fi

# ── Detect target repo language ──
# If --repo-root points to a non-Rust target, skip Rust-specific env vars.
_REPO_ROOT=""
_prev_arg=""
for arg in "$@"; do
    if [[ "$_prev_arg" == "--repo-root" ]]; then
        _REPO_ROOT="$arg"
    fi
    _prev_arg="$arg"
done
_TARGET_LANG="rust"
if [[ -n "$_REPO_ROOT" && -f "$_REPO_ROOT/.swarm/profile.toml" ]]; then
    _TARGET_LANG="$(grep -m1 '^language' "$_REPO_ROOT/.swarm/profile.toml" 2>/dev/null | sed 's/.*= *"\(.*\)"/\1/' || echo rust)"
fi

# ── sccache + shared target dir (Rust targets only) ──
if [[ "$_TARGET_LANG" == "rust" ]]; then
    if command -v sccache &>/dev/null; then
        export RUSTC_WRAPPER=sccache
        export SCCACHE_DIR="${SCCACHE_DIR:-/tmp/beefcake-sccache}"
        mkdir -p "$SCCACHE_DIR"
    fi
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/beefcake-shared-target}"
    mkdir -p "$CARGO_TARGET_DIR"
else
    echo "Non-Rust target (${_TARGET_LANG}): skipping RUSTC_WRAPPER and CARGO_TARGET_DIR"
fi

# ── Ensure PATH includes uv-installed tools (ruff, black, mypy, pytest) ──
export PATH="$HOME/.local/bin:$PATH"

# Use prebuilt release binary if available, otherwise fall back to cargo run.
# The release binary is 10x faster to start (no compilation) and runs faster.
# Build with: CARGO_TARGET_DIR=/tmp/beefcake-shared-target cargo build -p swarm-agents --release
_RELEASE_BIN="${CARGO_TARGET_DIR:-/tmp/beefcake-shared-target}/release/swarm-agents"
if [[ -x "$_RELEASE_BIN" ]]; then
    exec "$_RELEASE_BIN" "$@"
else
    exec cargo run -p swarm-agents -- "$@"
fi
