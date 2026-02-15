#!/bin/bash
#SBATCH --job-name=swarm-orch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/shared/ai/logs/swarm-orch-%j.log

REPO_DIR="/cluster/shared/code/beefcake-swarm"
WT_DIR="/cluster/shared/wt"
# Use node-local scratch to avoid NFS root_squash permission issues
CARGO_CACHE="/scratch/swarm-cargo-cache"
SIF="/cluster/shared/containers/swarm-agent.sif"

mkdir -p "$WT_DIR" "$CARGO_CACHE"

apptainer exec \
    --containall \
    --bind "${REPO_DIR}:/workspace" \
    --bind "${WT_DIR}:${WT_DIR}" \
    --bind "${CARGO_CACHE}:/opt/cargo-cache" \
    --pwd /workspace \
    --env "RUST_LOG=${RUST_LOG:-info}" \
    --env "SWARM_FAST_URL=http://vasp-02:8080/v1" \
    --env "SWARM_FAST_MODEL=Qwen3-Coder-Next-UD-Q4_K_XL.gguf" \
    --env "SWARM_CODER_URL=http://vasp-02:8080/v1" \
    --env "SWARM_CODER_MODEL=Qwen3-Coder-Next-UD-Q4_K_XL.gguf" \
    --env "SWARM_REASONING_URL=http://vasp-01:8081/v1" \
    --env "SWARM_REASONING_MODEL=or1-behemoth-q4_k_m.gguf" \
    --env "SWARM_CLOUD_URL=http://100.105.113.58:8317/v1" \
    --env "SWARM_CLOUD_API_KEY=rust-daq-proxy-key" \
    --env "SWARM_CLOUD_MODEL=claude-sonnet-4-5" \
    --env "SWARM_BEADS_BIN=bd" \
    "$SIF" \
    cargo run -p swarm-agents --release
