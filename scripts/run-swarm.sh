#!/usr/bin/env bash
set -euo pipefail
export RUST_LOG="${RUST_LOG:-info}"
# Worker tier: HydraCoder on vasp-02
export SWARM_WORKER_URL="${SWARM_WORKER_URL:-http://vasp-02:8080/v1}"
export SWARM_WORKER_MODEL="${SWARM_WORKER_MODEL:-HydraCoder-Q6_K.gguf}"
# Local manager: Qwen3.5 distributed on vasp-01+vasp-03
export SWARM_LOCAL_MANAGER_URL="${SWARM_LOCAL_MANAGER_URL:-http://vasp-01:8081/v1}"
export SWARM_LOCAL_MANAGER_MODEL="${SWARM_LOCAL_MANAGER_MODEL:-Qwen3.5-397B-A17B-UD-Q4_K_XL.gguf}"
# Cloud manager: Opus 4.5 via proxy
export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:-http://10.0.0.5:8317/v1}"
: "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
export SWARM_CLOUD_API_KEY
export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-opus-4-5-20250514}"
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-bd}"
exec cargo run -p swarm-agents "$@"
