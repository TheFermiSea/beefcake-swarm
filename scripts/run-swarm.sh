#!/usr/bin/env bash
set -euo pipefail
export RUST_LOG="${RUST_LOG:-info}"
export SWARM_FAST_URL="${SWARM_FAST_URL:-http://vasp-02:8080/v1}"
export SWARM_FAST_MODEL="${SWARM_FAST_MODEL:-Qwen3-Coder-Next-UD-Q4_K_XL}"
export SWARM_CODER_URL="${SWARM_CODER_URL:-http://vasp-02:8080/v1}"
export SWARM_CODER_MODEL="${SWARM_CODER_MODEL:-Qwen3-Coder-Next-UD-Q4_K_XL}"
export SWARM_REASONING_URL="${SWARM_REASONING_URL:-http://vasp-01:8081/v1}"
export SWARM_REASONING_MODEL="${SWARM_REASONING_MODEL:-or1-behemoth-q4_k_m.gguf}"
export SWARM_CLOUD_URL="${SWARM_CLOUD_URL:-http://10.0.0.5:8317/v1}"
: "${SWARM_CLOUD_API_KEY:?SWARM_CLOUD_API_KEY must be set}"
export SWARM_CLOUD_API_KEY
export SWARM_CLOUD_MODEL="${SWARM_CLOUD_MODEL:-claude-opus-4-6-thinking}"
export SWARM_BEADS_BIN="${SWARM_BEADS_BIN:-bd}"
exec cargo run -p swarm-agents "$@"
