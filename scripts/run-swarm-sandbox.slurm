#!/bin/bash
# run-swarm-sandbox.slurm — SLURM job script for running the swarm orchestrator
# inside an Apptainer container on the HPC cluster.
#
# Usage:
#   sbatch scripts/run-swarm-sandbox.slurm                        # pick from beads backlog
#   sbatch --export=SWARM_ISSUE_ID=foo scripts/run-swarm-sandbox.slurm  # specific issue
#
# Required env vars (via sbatch --export or inherited):
#   SWARM_CLOUD_API_KEY   — CLIAPIProxy API key (or file at SWARM_CLOUD_API_KEY_FILE)
#
# All SWARM_* env vars are forwarded into the container. See CLAUDE.md for the
# full reference. Defaults here match the current cluster layout (Apr 2026).
#
#SBATCH --job-name=swarm-orch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/shared/ai/logs/swarm-orch-%j.log
#SBATCH --uid=brian
#SBATCH --gid=hpc

set -euo pipefail

# ── Configurable paths (override via env) ──
REPO_DIR="${SWARM_SANDBOX_REPO_DIR:-/cluster/shared/code/beefcake-swarm}"
WT_DIR="${SWARM_SANDBOX_WT_DIR:-/cluster/shared/wt}"
# Use node-local scratch to avoid NFS root_squash permission issues
CARGO_CACHE="${SWARM_SANDBOX_CARGO_CACHE:-/scratch/swarm-cargo-cache}"
CARGO_TARGET="${SWARM_SANDBOX_CARGO_TARGET:-/scratch/swarm-target}"
SIF="${SWARM_SANDBOX_SIF:-/cluster/shared/containers/swarm-agent.sif}"
CLOUD_PROXY_URL="${SWARM_CLOUD_URL:-http://10.0.0.5:8317/v1}"
CLOUD_API_KEY_FILE="${SWARM_CLOUD_API_KEY_FILE:-/cluster/shared/ai/.cloud-api-key}"

# ── Validate critical paths ──
if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 1
fi
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Apptainer SIF not found: ${SIF}" >&2
    exit 1
fi

mkdir -p "$WT_DIR" "$CARGO_CACHE" "$CARGO_TARGET"

# ── Temp file cleanup on exit ──
TMPFILES=()
cleanup() {
    for f in "${TMPFILES[@]}"; do
        rm -f "$f" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ── Model / endpoint defaults (match CLAUDE.md Apr 2026) ──
# Scout/Fast: GLM-4.7-Flash on vasp-03
FAST_URL="${SWARM_FAST_URL:-http://vasp-03:8081/v1}"
FAST_MODEL="${SWARM_FAST_MODEL:-GLM-4.7-Flash}"
# Coder: Qwen3.5-27B on vasp-01
CODER_URL="${SWARM_CODER_URL:-http://vasp-01:8081/v1}"
CODER_MODEL="${SWARM_CODER_MODEL:-Qwen3.5-27B}"
# Reasoning: Devstral-Small-2-24B on vasp-02
REASONING_URL="${SWARM_REASONING_URL:-http://vasp-02:8081/v1}"
REASONING_MODEL="${SWARM_REASONING_MODEL:-Devstral-Small-2-24B}"

# ── Worker fallback: if fast endpoint is down, route everything to reasoning ──
if [[ "${SWARM_WORKER_FALLBACK_REASONING:-1}" == "1" ]]; then
    if ! curl -fsS --max-time 3 "${FAST_URL%/v1}/health" >/dev/null 2>&1; then
        echo "Worker endpoint unavailable (${FAST_URL}); falling back to reasoning endpoint"
        FAST_URL="$REASONING_URL"
        FAST_MODEL="$REASONING_MODEL"
        CODER_URL="$REASONING_URL"
        CODER_MODEL="$REASONING_MODEL"
    fi
fi

# ── Recover stale worktree artifacts for this issue ──
if [[ -n "${SWARM_ISSUE_ID:-}" ]]; then
    # Validate issue ID: strict allowlist to prevent path traversal and injection
    if [[ ! "$SWARM_ISSUE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$SWARM_ISSUE_ID" == "." ]] || [[ "$SWARM_ISSUE_ID" == ".." ]]; then
        echo "ERROR: SWARM_ISSUE_ID contains invalid characters: ${SWARM_ISSUE_ID}" >&2
        exit 1
    fi
    ISSUE_SANITIZED="${SWARM_ISSUE_ID//./_}"
    ISSUE_WT_RAW="${WT_DIR}/${SWARM_ISSUE_ID}"
    ISSUE_WT_SANITIZED="${WT_DIR}/${ISSUE_SANITIZED}"
    git -C "$REPO_DIR" worktree remove --force "$ISSUE_WT_RAW" 2>/dev/null || true
    git -C "$REPO_DIR" worktree remove --force "$ISSUE_WT_SANITIZED" 2>/dev/null || true
    rm -rf "$ISSUE_WT_RAW" "$ISSUE_WT_SANITIZED" 2>/dev/null || true
    git -C "$REPO_DIR" branch -D "swarm/${SWARM_ISSUE_ID}" 2>/dev/null || true
    git -C "$REPO_DIR" branch -D "swarm/${ISSUE_SANITIZED}" 2>/dev/null || true
    git -C "$REPO_DIR" worktree prune 2>/dev/null || true
fi

# ── Cloud manager configuration ──
# Set SWARM_USE_CLOUD=0 to force local-only manager mode.
SWARM_USE_CLOUD="${SWARM_USE_CLOUD:-1}"
if [[ "$SWARM_USE_CLOUD" == "1" ]]; then
    SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-opus-4-6}"
    SWARM_CLOUD_FALLBACK_MODEL="${SWARM_CLOUD_FALLBACK_MODEL:-claude-sonnet-4-6}"

    # Load cloud API key from file if not set via environment
    if [[ -z "${SWARM_CLOUD_API_KEY:-}" ]] && [[ -f "$CLOUD_API_KEY_FILE" ]]; then
        SWARM_CLOUD_API_KEY="$(cat "$CLOUD_API_KEY_FILE")"
        export SWARM_CLOUD_API_KEY
    fi

    # CLIAPIProxy v6.8+ uses x-api-key header (not Authorization: Bearer)
    _PROXY_AUTH=(-H "x-api-key: ${SWARM_CLOUD_API_KEY:-}")

    # ── Ownership check ──
    if [[ "${SWARM_REQUIRE_ANTHROPIC_OWNERSHIP:-1}" == "1" ]] && [[ -n "${SWARM_CLOUD_API_KEY:-}" ]]; then
        MODELS_RESP="$(mktemp)"
        TMPFILES+=("$MODELS_RESP")
        if curl -sS "${_PROXY_AUTH[@]}" \
            "${CLOUD_PROXY_URL%/v1}/v1/models" > "$MODELS_RESP" 2>/dev/null; then
            MODEL_OWNER="$(python3 - "$MODELS_RESP" "$SWARM_CLOUD_MODEL" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
model = sys.argv[2]
entry = next((m for m in doc.get("data", []) if m.get("id") == model), None)
print((entry or {}).get("owned_by", ""))
PY
)"
            # Accept anthropic, antigravity (CLIAPIProxy alias), and openai
            if [[ -n "$MODEL_OWNER" && "$MODEL_OWNER" != "anthropic" && "$MODEL_OWNER" != "antigravity" && "$MODEL_OWNER" != "openai" ]]; then
                echo "Cloud model ${SWARM_CLOUD_MODEL} is owned_by=${MODEL_OWNER}; falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
                SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
            fi
        fi
        rm -f "$MODELS_RESP"
    fi

    # ── Preflight probe ──
    if [[ "${SWARM_CLOUD_PREFLIGHT:-1}" == "1" ]] && [[ -n "${SWARM_CLOUD_API_KEY:-}" ]]; then
        PROBE_REQ="$(mktemp)"
        PROBE_RESP="${PROBE_REQ}.out"
        TMPFILES+=("$PROBE_REQ" "$PROBE_RESP")
        printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}\n' \
            "$SWARM_CLOUD_MODEL" > "$PROBE_REQ"
        PROBE_HTTP="$(curl -sS -o "$PROBE_RESP" -w "%{http_code}" \
            "${_PROXY_AUTH[@]}" \
            -H "Content-Type: application/json" \
            "${CLOUD_PROXY_URL%/v1}/v1/chat/completions" \
            -d @"$PROBE_REQ" || echo "000")"
        if [[ "$PROBE_HTTP" != "200" ]] || grep -qiE 'auth_unavailable|quota_exhausted|resource_exhausted|exhausted your capacity|quota will reset' "$PROBE_RESP"; then
            echo "Cloud model ${SWARM_CLOUD_MODEL} unavailable (http=${PROBE_HTTP}); falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
            SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
        fi
        rm -f "$PROBE_REQ" "$PROBE_RESP"
    fi
fi

# Fix .git object permissions so worktrees can write to the ODB.
# Required because a previous `git pull` as root may have locked these.
chown -R brian:hpc "${REPO_DIR}/.git" 2>/dev/null || true

# bd v0.52.0 (non-CGO) requires dolt which isn't in the container.
# Use a lightweight Python shim that reads/writes .beads/issues.jsonl directly.
BD_SHIM="${REPO_DIR}/scripts/bd-jsonl-shim.py"
if [[ -f "$BD_SHIM" ]]; then
    chmod +x "$BD_SHIM"
    echo "Using bd-jsonl-shim for beads operations"
fi

# ── Build container env args ──
CLOUD_ENV_ARGS=()
if [[ "$SWARM_USE_CLOUD" == "1" ]]; then
    CLOUD_ENV_ARGS+=(
        --env "SWARM_CLOUD_URL=${CLOUD_PROXY_URL}"
        --env "SWARM_CLOUD_API_KEY=${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
        --env "SWARM_CLOUD_MODEL=${SWARM_CLOUD_MODEL}"
        --env "SWARM_CLOUD_FALLBACK_MODEL=${SWARM_CLOUD_FALLBACK_MODEL}"
    )
else
    echo "SWARM_USE_CLOUD=0: using local manager mode (no cloud endpoint)"
fi

# Validator models (for multi-model validation gate)
VALIDATOR_MODEL_1="${SWARM_VALIDATOR_MODEL_1:-gemini-3-pro-preview}"
VALIDATOR_MODEL_2="${SWARM_VALIDATOR_MODEL_2:-claude-sonnet-4-6}"

# --cleanenv: clean environment (only explicit --env vars)
# --writable-tmpfs: allows writes to SIF overlay areas
# Explicit --bind for writable paths so Apptainer doesn't overlay them
apptainer exec \
    --cleanenv \
    --writable-tmpfs \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    --bind "${WT_DIR}:${WT_DIR}" \
    --bind "${CARGO_CACHE}:${CARGO_CACHE}" \
    --bind "${CARGO_TARGET}:${CARGO_TARGET}" \
    --bind "/cluster/shared/tools:/cluster/shared/tools" \
    --pwd "${REPO_DIR}" \
    --env "RUST_LOG=${RUST_LOG:-swarm_agents=info,coordination=info,rig=debug,rig_core=info}" \
    --env "CARGO_HOME=${CARGO_CACHE}" \
    --env "CARGO_TARGET_DIR=${CARGO_TARGET}" \
    --env "SWARM_FAST_URL=${FAST_URL}" \
    --env "SWARM_FAST_MODEL=${FAST_MODEL}" \
    --env "SWARM_CODER_URL=${CODER_URL}" \
    --env "SWARM_CODER_MODEL=${CODER_MODEL}" \
    --env "SWARM_REASONING_URL=${REASONING_URL}" \
    --env "SWARM_REASONING_MODEL=${REASONING_MODEL}" \
    "${CLOUD_ENV_ARGS[@]}" \
    --env "SWARM_MANAGER_MAX_TURNS=${SWARM_MANAGER_MAX_TURNS:-40}" \
    --env "SWARM_MAX_RETRIES=${SWARM_MAX_RETRIES:-20}" \
    --env "SWARM_MAX_NO_CHANGE=${SWARM_MAX_NO_CHANGE:-3}" \
    --env "SWARM_COUNCIL_MAX_ITERATIONS=${SWARM_COUNCIL_MAX_ITERATIONS:-12}" \
    --env "SWARM_COUNCIL_MAX_CONSULTATIONS=${SWARM_COUNCIL_MAX_CONSULTATIONS:-12}" \
    --env "SWARM_CLOUD_MAX_RETRIES=${SWARM_CLOUD_MAX_RETRIES:-6}" \
    --env "SWARM_INITIAL_TIER=${SWARM_INITIAL_TIER:-council}" \
    --env "SWARM_MANAGER_TIMEOUT_SECS=${SWARM_MANAGER_TIMEOUT_SECS:-300}" \
    --env "SWARM_WORKER_TIMEOUT_SECS=${SWARM_WORKER_TIMEOUT_SECS:-1800}" \
    --env "SWARM_VALIDATION_TIMEOUT_SECS=${SWARM_VALIDATION_TIMEOUT_SECS:-180}" \
    --env "SWARM_CONTEXT_TOKENS_WORKER=${SWARM_CONTEXT_TOKENS_WORKER:-8000}" \
    --env "SWARM_CONTEXT_TOKENS_COUNCIL=${SWARM_CONTEXT_TOKENS_COUNCIL:-12000}" \
    --env "SWARM_CONTEXT_TOKENS_HUMAN=${SWARM_CONTEXT_TOKENS_HUMAN:-32000}" \
    --env "SWARM_ISSUE_ID=${SWARM_ISSUE_ID:-}" \
    --env "SWARM_ISSUE_OBJECTIVE=${SWARM_ISSUE_OBJECTIVE:-}" \
    --env "SWARM_ISSUE_OBJECTIVE_B64=${SWARM_ISSUE_OBJECTIVE_B64:-}" \
    --env "SWARM_VALIDATOR_MODEL_1=${VALIDATOR_MODEL_1}" \
    --env "SWARM_VALIDATOR_MODEL_2=${VALIDATOR_MODEL_2}" \
    --env "SWARM_BEADS_BIN=${REPO_DIR}/scripts/bd-jsonl-shim.py" \
    --env "SWARM_NLM_BIN=/cluster/shared/tools/nlm/bin/nlm" \
    --env "NOTEBOOKLM_MCP_CLI_PATH=/cluster/shared/tools/nlm/config" \
    --env "SWARM_TENSORZERO_URL=${SWARM_TENSORZERO_URL:-}" \
    --env "SWARM_TENSORZERO_PG_URL=${SWARM_TENSORZERO_PG_URL:-}" \
    --env "SWARM_REPO_ID=${SWARM_REPO_ID:-beefcake-swarm}" \
    --env "BD_ACTOR=${BD_ACTOR:-swarm-sandbox-${SLURM_JOB_ID:-unknown}}" \
    --env "GIT_AUTHOR_NAME=Swarm Agent" \
    --env "GIT_COMMITTER_NAME=Swarm Agent" \
    --env "GIT_AUTHOR_EMAIL=swarm@beefcake.local" \
    --env "GIT_COMMITTER_EMAIL=swarm@beefcake.local" \
    --env "PATH=/cluster/shared/tools/nlm/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$SIF" \
    bash -c '
      OBJ="See beads issue ${SWARM_ISSUE_ID} for full spec"
      if [ -n "${SWARM_ISSUE_OBJECTIVE_B64:-}" ]; then
          OBJ="$(printf "%s" "$SWARM_ISSUE_OBJECTIVE_B64" | base64 -d 2>/dev/null || printf "%s" "$OBJ")"
      elif [ -n "${SWARM_ISSUE_OBJECTIVE:-}" ]; then
          OBJ="${SWARM_ISSUE_OBJECTIVE}"
      fi
      ISSUE_ARGS=()
      if [ -n "${SWARM_ISSUE_ID:-}" ]; then
          ISSUE_ARGS+=(--issue "$SWARM_ISSUE_ID" --objective "$OBJ")
      fi
      stdbuf -oL -eL cargo run -p swarm-agents --release -- "${ISSUE_ARGS[@]}"
    '
