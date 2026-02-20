#!/bin/bash
#SBATCH --job-name=swarm-orch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/shared/ai/logs/swarm-orch-%j.log
#SBATCH --uid=brian
#SBATCH --gid=hpc

REPO_DIR="/cluster/shared/code/beefcake-swarm"
WT_DIR="/cluster/shared/wt"
# Use node-local scratch to avoid NFS root_squash permission issues
CARGO_CACHE="/scratch/swarm-cargo-cache"
CARGO_TARGET="/scratch/swarm-target"
SIF="/cluster/shared/containers/swarm-agent.sif"

mkdir -p "$WT_DIR" "$CARGO_CACHE" "$CARGO_TARGET"

FAST_URL="${SWARM_FAST_URL:-http://vasp-02:8080/v1}"
FAST_MODEL="${SWARM_FAST_MODEL:-strand-rust-coder-14b-q8_0}"
CODER_URL="${SWARM_CODER_URL:-http://vasp-02:8080/v1}"
CODER_MODEL="${SWARM_CODER_MODEL:-HydraCoder.i1-Q4_K_M}"
REASONING_URL="${SWARM_REASONING_URL:-http://vasp-01:8081/v1}"
REASONING_MODEL="${SWARM_REASONING_MODEL:-or1-behemoth-q4_k_m.gguf}"

# Temporary failover during worker endpoint maintenance (e.g., vasp-02 MPS setup).
if [ "${SWARM_WORKER_FALLBACK_REASONING:-1}" = "1" ]; then
    if ! curl -fsS --max-time 3 "${FAST_URL%/v1}/health" >/dev/null 2>&1; then
        echo "Worker endpoint unavailable (${FAST_URL}); falling back to reasoning endpoint"
        FAST_URL="$REASONING_URL"
        FAST_MODEL="$REASONING_MODEL"
        CODER_URL="$REASONING_URL"
        CODER_MODEL="$REASONING_MODEL"
    fi
fi

# Recover from stale worktree artifacts for this issue (if any).
if [ -n "${SWARM_ISSUE_ID:-}" ]; then
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

# Set SWARM_USE_CLOUD=0 to force local OR1 manager mode.
SWARM_USE_CLOUD="${SWARM_USE_CLOUD:-1}"
if [ "$SWARM_USE_CLOUD" = "1" ]; then
    SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-sonnet-4-6}"
    SWARM_CLOUD_FALLBACK_MODEL="${SWARM_CLOUD_FALLBACK_MODEL:-claude-sonnet-4-5-20250929}"
    # Load cloud API key from file if not set via environment
    if [ -z "${SWARM_CLOUD_API_KEY:-}" ] && [ -f /cluster/shared/ai/.cloud-api-key ]; then
        SWARM_CLOUD_API_KEY="$(cat /cluster/shared/ai/.cloud-api-key)"
        export SWARM_CLOUD_API_KEY
    fi
    if [ "${SWARM_REQUIRE_ANTHROPIC_OWNERSHIP:-1}" = "1" ] && [ -n "${SWARM_CLOUD_API_KEY:-}" ]; then
        MODELS_RESP="/tmp/swarm-cloud-models-$$.json"
        if curl -sS -H "Authorization: Bearer $SWARM_CLOUD_API_KEY" \
            "http://10.0.0.5:8317/v1/models" > "$MODELS_RESP"; then
            MODEL_OWNER="$(python3 - "$MODELS_RESP" "$SWARM_CLOUD_MODEL" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
model = sys.argv[2]
entry = next((m for m in doc.get("data", []) if m.get("id") == model), None)
print((entry or {}).get("owned_by", ""))
PY
)"
            if [ -n "$MODEL_OWNER" ] && [ "$MODEL_OWNER" != "anthropic" ]; then
                echo "Cloud model ${SWARM_CLOUD_MODEL} is owned_by=${MODEL_OWNER}; falling back to ${SWARM_CLOUD_FALLBACK_MODEL}"
                SWARM_CLOUD_MODEL="$SWARM_CLOUD_FALLBACK_MODEL"
            fi
        fi
        rm -f "$MODELS_RESP"
    fi
    # Preflight current manager model and auto-fallback if quota/auth is unavailable.
    if [ "${SWARM_CLOUD_PREFLIGHT:-1}" = "1" ] && [ -n "${SWARM_CLOUD_API_KEY:-}" ]; then
        PROBE_REQ="/tmp/swarm-cloud-probe-$$.json"
        PROBE_RESP="${PROBE_REQ}.out"
        printf '{"model":"%s","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}\n' \
            "$SWARM_CLOUD_MODEL" > "$PROBE_REQ"
        PROBE_HTTP="$(curl -sS -o "$PROBE_RESP" -w "%{http_code}" \
            -H "Authorization: Bearer $SWARM_CLOUD_API_KEY" \
            -H "Content-Type: application/json" \
            "http://10.0.0.5:8317/v1/chat/completions" \
            -d @"$PROBE_REQ" || echo "000")"
        if [ "$PROBE_HTTP" != "200" ] || grep -qiE 'auth_unavailable|quota_exhausted|resource_exhausted|exhausted your capacity|quota will reset' "$PROBE_RESP"; then
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
if [ -f "$BD_SHIM" ]; then
    chmod +x "$BD_SHIM"
    echo "Using bd-jsonl-shim for beads operations"
fi

# Build cloud env args only when enabled.
CLOUD_ENV_ARGS=()
if [ "$SWARM_USE_CLOUD" = "1" ]; then
    CLOUD_ENV_ARGS+=(
        --env "SWARM_CLOUD_URL=http://10.0.0.5:8317/v1"
        --env "SWARM_CLOUD_API_KEY=${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
        --env "SWARM_CLOUD_MODEL=${SWARM_CLOUD_MODEL}"
    )
else
    echo "SWARM_USE_CLOUD=0: using local OR1 manager (no cloud endpoint)"
fi

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
    --env "SWARM_VALIDATOR_MODEL_1=gemini-3-pro-preview" \
    --env "SWARM_VALIDATOR_MODEL_2=claude-sonnet-4-5-20250929" \
    --env "SWARM_BEADS_BIN=${REPO_DIR}/scripts/bd-jsonl-shim.py" \
    --env "SWARM_NLM_BIN=/cluster/shared/tools/nlm/bin/nlm" \
    --env "NOTEBOOKLM_MCP_CLI_PATH=/cluster/shared/tools/nlm/config" \
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
