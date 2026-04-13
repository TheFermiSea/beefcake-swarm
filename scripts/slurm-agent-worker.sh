#!/bin/bash
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@30
#SBATCH --uid=brian
#SBATCH --gid=hpc

###############################################################################
# slurm-agent-worker.sh — Run a swarm-agents worker ON a compute node.
#
# This runs as a SLURM job on vasp-01/02/03. The key advantage:
# each node already has an inference server on localhost:8081, so the
# agent talks to its local model with zero network overhead.
#
# Cross-tier calls still reach other nodes by hostname (vasp-01/02/03:8081).
# Cloud manager calls go to ai-proxy via CLIAPIProxy at 10.0.0.100:8317.
#
# Node → Model mapping:
#   vasp-01 → Qwen3.5-27B (coder, SWARM_CODER_URL=localhost)
#   vasp-02 → Devstral-Small-2-24B (reasoning, SWARM_REASONING_URL=localhost)
#   vasp-03 → GLM-4.7-Flash (scout/fast, SWARM_FAST_URL=localhost)
#
# Environment (set by slurm-agent-dispatch.sh via --export):
#   ISSUE_ID             — Beads issue ID to work on
#   ISSUE_OBJECTIVE_B64  — Base64-encoded issue title/objective
#   DISPATCH_NODE        — Target node name (vasp-01, vasp-02, vasp-03)
#
# The script auto-detects which node it runs on and overrides the
# corresponding tier URL to localhost, keeping other tiers as remote.
###############################################################################

set -euo pipefail

ISSUE_ID="${ISSUE_ID:?ISSUE_ID must be set}"
REPO_ROOT="${REPO_ROOT:-/home/brian/code/beefcake-swarm}"
WORKER_DIR="/tmp/beefcake-wt/${ISSUE_ID}"
NODE_NAME="$(hostname -s)"

# Decode objective
OBJECTIVE="See beads issue ${ISSUE_ID} for full spec"
if [[ -n "${ISSUE_OBJECTIVE_B64:-}" ]]; then
    OBJECTIVE="$(printf "%s" "$ISSUE_OBJECTIVE_B64" | base64 -d 2>/dev/null || echo "$OBJECTIVE")"
fi

echo "==========================================="
echo "Swarm Agent Worker"
echo "==========================================="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${NODE_NAME}"
echo "Issue:      ${ISSUE_ID}"
echo "Objective:  ${OBJECTIVE}"
echo "Repo:       ${REPO_ROOT}"
echo "Worktree:   ${WORKER_DIR}"
echo "==========================================="

# ── Graceful shutdown ──
WORKER_PID=""
SHUTTING_DOWN=false

cleanup() {
    if $SHUTTING_DOWN; then return; fi
    SHUTTING_DOWN=true

    echo "[$(date)] SIGTERM received — shutting down worker"

    if [[ -n "$WORKER_PID" ]]; then
        kill -TERM "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi

    # Attempt to salvage any uncommitted changes before removing worktree
    if [[ -d "$WORKER_DIR" ]]; then
        cd "$WORKER_DIR"
        if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
            echo "[$(date)] No uncommitted changes to salvage"
        else
            echo "[$(date)] Salvaging uncommitted changes..."
            git add -A 2>/dev/null || true
            git commit -m "salvage: uncommitted work from SLURM job ${SLURM_JOB_ID:-unknown} (SIGTERM)" 2>/dev/null || true
        fi
    fi

    # Clean up worktree
    cd "$REPO_ROOT" 2>/dev/null || true
    git worktree remove "$WORKER_DIR" --force 2>/dev/null || true
    git worktree prune 2>/dev/null || true

    echo "[$(date)] Cleanup complete"
}
trap cleanup SIGTERM SIGINT EXIT

# ── Validate prerequisites ──
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    echo "ERROR: Repo not found at $REPO_ROOT" >&2
    exit 1
fi

# Validate issue ID: strict allowlist to prevent path traversal
if [[ ! "$ISSUE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$ISSUE_ID" == "." ]] || [[ "$ISSUE_ID" == ".." ]]; then
    echo "ERROR: ISSUE_ID contains invalid characters: ${ISSUE_ID}" >&2
    exit 1
fi

# Check that local inference is running
if ! curl -sf --max-time 5 http://localhost:8081/health &>/dev/null; then
    echo "WARNING: Local inference server not responding on localhost:8081"
    echo "  The agent will fall back to remote endpoints."
fi

# ── Node-aware model routing ──
# Override the tier URL for the model running locally on this node.
# Keep other tiers as remote hostnames so cross-tier calls still work.
export SWARM_FAST_URL="${SWARM_FAST_URL:-http://vasp-03:8081/v1}"
export SWARM_FAST_MODEL="${SWARM_FAST_MODEL:-GLM-4.7-Flash}"
export SWARM_CODER_URL="${SWARM_CODER_URL:-http://vasp-01:8081/v1}"
export SWARM_CODER_MODEL="${SWARM_CODER_MODEL:-Qwen3.5-27B}"
export SWARM_REASONING_URL="${SWARM_REASONING_URL:-http://vasp-02:8081/v1}"
export SWARM_REASONING_MODEL="${SWARM_REASONING_MODEL:-Qwen3.5-27B}"

case "$NODE_NAME" in
    vasp-01*)
        echo "[routing] vasp-01: Qwen3.5-27B local → SWARM_CODER_URL=localhost"
        export SWARM_CODER_URL="http://localhost:8081/v1"
        ;;
    vasp-02*)
        echo "[routing] vasp-02: Devstral-Small-2-24B local → SWARM_REASONING_URL=localhost"
        export SWARM_REASONING_URL="http://localhost:8081/v1"
        ;;
    vasp-03*)
        echo "[routing] vasp-03: GLM-4.7-Flash local → SWARM_FAST_URL=localhost"
        export SWARM_FAST_URL="http://localhost:8081/v1"
        ;;
    *)
        echo "[routing] Unknown node ${NODE_NAME}: using all remote endpoints"
        ;;
esac

# ── Cloud proxy (reachable from compute nodes via ai-proxy LAN IP) ──
# ai-proxy's CLIAPIProxy is at 10.0.0.100:8317 from the cluster network.
# Only set if not already configured via environment.
if [[ -z "${SWARM_CLOUD_URL:-}" ]]; then
    export SWARM_CLOUD_URL="http://10.0.0.100:8317/v1"
fi
# Load cloud API key from shared NFS if not in environment
if [[ -z "${SWARM_CLOUD_API_KEY:-}" && -f /cluster/shared/ai/.cloud-api-key ]]; then
    SWARM_CLOUD_API_KEY="$(cat /cluster/shared/ai/.cloud-api-key)"
    export SWARM_CLOUD_API_KEY
fi

# ── Beads ──
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-bd}"
export BD_ACTOR="${BD_ACTOR:-worker-${NODE_NAME}-${ISSUE_ID}}"

# ── Git identity for commits from compute nodes ──
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-Swarm Agent}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-Swarm Agent}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-swarm@beefcake.local}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-swarm@beefcake.local}"

# ── Logging ──
export RUST_LOG="${RUST_LOG:-info}"

# ── sccache + shared target dir (NFS for cross-node binary sharing) ──
if command -v sccache &>/dev/null; then
    export RUSTC_WRAPPER=sccache
    export SCCACHE_DIR="${SCCACHE_DIR:-/tmp/beefcake-sccache}"
    mkdir -p "$SCCACHE_DIR"
fi
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cluster/shared/cargo-cache/beefcake-target}"
mkdir -p "$CARGO_TARGET_DIR" 2>/dev/null || true

# ── PATH (brian's NFS-mounted cargo + local tools) ──
if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ── Create worktree ──
cd "$REPO_ROOT"

# Clean up any stale worktree from a previous run of this issue
git worktree remove "$WORKER_DIR" --force 2>/dev/null || true
rm -rf "$WORKER_DIR" 2>/dev/null || true
git branch -D "swarm/${ISSUE_ID}" 2>/dev/null || true
git worktree prune 2>/dev/null || true

# Ensure we're up to date
git fetch origin main --quiet 2>/dev/null || true

# Create fresh worktree
if ! git worktree add "$WORKER_DIR" -b "swarm/${ISSUE_ID}" origin/main 2>/dev/null; then
    # Branch might exist from a failed run — try checking it out
    if ! git worktree add "$WORKER_DIR" "swarm/${ISSUE_ID}" 2>/dev/null; then
        echo "ERROR: Failed to create worktree at $WORKER_DIR" >&2
        exit 1
    fi
fi

cd "$WORKER_DIR"
echo "[worker] Worktree created at $WORKER_DIR (branch: swarm/${ISSUE_ID})"

# ── Run the agent ──
# Use prebuilt release binary if available, otherwise fall back to cargo run.
# Search order: NFS shared binary → cargo target dir → cargo run from source.
_RELEASE_BIN=""
for _candidate in \
    "/cluster/shared/ai/bin/swarm-agents" \
    "${CARGO_TARGET_DIR}/release/swarm-agents"; do
    if [[ -x "$_candidate" ]]; then
        _RELEASE_BIN="$_candidate"
        break
    fi
done

echo "[worker] Starting swarm-agents for issue ${ISSUE_ID}..."

if [[ -n "$_RELEASE_BIN" ]]; then
    echo "[worker] Using release binary: $_RELEASE_BIN"
    timeout "${SWARM_SUBTASK_TIMEOUT_SECS:-3600}" "$_RELEASE_BIN" \
        --issue "$ISSUE_ID" \
        --objective "$OBJECTIVE" \
        2>&1 &
else
    echo "[worker] No release binary found; using cargo run"
    timeout "${SWARM_SUBTASK_TIMEOUT_SECS:-3600}" cargo run -p swarm-agents --release -- \
        --issue "$ISSUE_ID" \
        --objective "$OBJECTIVE" \
        2>&1 &
fi
WORKER_PID=$!

echo "[worker] Agent PID: $WORKER_PID"
wait "$WORKER_PID"
EXIT_CODE=$?
WORKER_PID=""

echo "[worker] Agent exited with code $EXIT_CODE"

# ── Post-run: push branch if there are commits ──
if git log "origin/main..HEAD" --oneline 2>/dev/null | head -1 | grep -q .; then
    echo "[worker] Pushing branch swarm/${ISSUE_ID}..."
    git push origin "swarm/${ISSUE_ID}" --force-with-lease 2>/dev/null || \
        echo "WARNING: Failed to push branch (manual merge needed)"
else
    echo "[worker] No new commits on branch"
fi

# Cleanup happens in the EXIT trap
exit $EXIT_CODE
