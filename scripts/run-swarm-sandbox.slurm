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

# Load cloud API key from file if not set via environment
if [ -z "${SWARM_CLOUD_API_KEY:-}" ] && [ -f /cluster/shared/ai/.cloud-api-key ]; then
    SWARM_CLOUD_API_KEY="$(cat /cluster/shared/ai/.cloud-api-key)"
    export SWARM_CLOUD_API_KEY
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
    --env "SWARM_FAST_URL=http://vasp-02:8080/v1" \
    --env "SWARM_FAST_MODEL=Qwen3-Coder-Next-UD-Q4_K_XL" \
    --env "SWARM_CODER_URL=http://vasp-02:8080/v1" \
    --env "SWARM_CODER_MODEL=Qwen3-Coder-Next-UD-Q4_K_XL" \
    --env "SWARM_REASONING_URL=http://vasp-01:8081/v1" \
    --env "SWARM_REASONING_MODEL=or1-behemoth-q4_k_m.gguf" \
    --env "SWARM_CLOUD_URL=http://10.0.0.5:8317/v1" \
    --env "SWARM_CLOUD_API_KEY=${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}" \
    --env "SWARM_CLOUD_MODEL=claude-opus-4-5-20251101" \
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
    stdbuf -oL -eL cargo run -p swarm-agents --release
