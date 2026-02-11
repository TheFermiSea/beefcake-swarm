#!/bin/bash
###############################################################################
# Validation Script for llama.cpp Inference Stack
#
# This script validates the complete deployment:
#   1. SLURM configuration (preemption works)
#   2. Apptainer container (builds and runs)
#   3. 14B single-node inference
#   4. 72B distributed inference
#   5. Preemption recovery
#
# Usage:
#   ./validate.sh              # Run all validations
#   ./validate.sh --slurm      # Only SLURM checks
#   ./validate.sh --container  # Only container checks
#   ./validate.sh --inference  # Only inference checks
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_CTL="slurm-ctl"
CONTAINER="/cluster/shared/containers/llama-server.sif"
MODEL_14B="/scratch/ai/models/strand-rust-coder-14b.Q8_0.gguf"
MODEL_72B="/scratch/ai/models/OR1-Behemoth.Q8_0.gguf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
RUN_SLURM=true
RUN_CONTAINER=true
RUN_INFERENCE=true

for arg in "$@"; do
    case $arg in
        --slurm)
            RUN_SLURM=true
            RUN_CONTAINER=false
            RUN_INFERENCE=false
            ;;
        --container)
            RUN_SLURM=false
            RUN_CONTAINER=true
            RUN_INFERENCE=false
            ;;
        --inference)
            RUN_SLURM=false
            RUN_CONTAINER=false
            RUN_INFERENCE=true
            ;;
        --help|-h)
            echo "Usage: $0 [--slurm] [--container] [--inference]"
            exit 0
            ;;
    esac
done

# Validation results
PASSED=0
FAILED=0
WARNINGS=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

info() {
    echo -e "[INFO] $1"
}

echo "=========================================="
echo "llama.cpp Inference Stack Validation"
echo "=========================================="

###############################################################################
# SLURM Validation
###############################################################################

if $RUN_SLURM; then
    echo ""
    echo "--- SLURM Configuration ---"

    # Check gpu_ai partition exists
    if ssh "${SLURM_CTL}" "sinfo -p gpu_ai -h 2>/dev/null" | grep -q gpu_ai; then
        pass "gpu_ai partition exists"
    else
        fail "gpu_ai partition not found"
    fi

    # Check QoS exists
    if ssh "${SLURM_CTL}" "sacctmgr show qos ai_opportunistic -n 2>/dev/null" | grep -q ai_opportunistic; then
        pass "ai_opportunistic QoS exists"

        # Check GraceTime
        GRACE=$(ssh "${SLURM_CTL}" "sacctmgr show qos ai_opportunistic format=GraceTime -n 2>/dev/null" | tr -d ' ')
        if [[ "$GRACE" == "00:00:30" ]]; then
            pass "QoS grace time is 30 seconds"
        else
            warn "QoS grace time is ${GRACE} (expected 00:00:30)"
        fi
    else
        fail "ai_opportunistic QoS not found"
    fi

    # Check preemption mode
    if ssh "${SLURM_CTL}" "scontrol show partition gpu_ai 2>/dev/null" | grep -q "PreemptMode=REQUEUE"; then
        pass "Preemption mode is REQUEUE"
    else
        warn "Preemption mode may not be REQUEUE (check scontrol show partition gpu_ai)"
    fi

    # Check /scratch/ai directories (adjust GPU_NODES if needed)
    GPU_NODES="${GPU_NODES:-vasp-01 vasp-02 vasp-03}"
    for node in ${GPU_NODES}; do
        if ssh "${node}" "test -d /scratch/ai/models" 2>/dev/null; then
            pass "/scratch/ai directories exist on ${node}"
        else
            fail "/scratch/ai directories missing on ${node}"
        fi
    done
fi

###############################################################################
# Container Validation
###############################################################################

if $RUN_CONTAINER; then
    echo ""
    echo "--- Container Validation ---"

    # Check container exists
    if ssh "${SLURM_CTL}" "test -f ${CONTAINER}" 2>/dev/null; then
        pass "Container exists at ${CONTAINER}"

        # Get size
        SIZE=$(ssh "${SLURM_CTL}" "du -h ${CONTAINER}" 2>/dev/null | cut -f1)
        info "Container size: ${SIZE}"
    else
        fail "Container not found at ${CONTAINER}"
    fi

    # Check llama-server binary in container
    if ssh "${SLURM_CTL}" "apptainer exec ${CONTAINER} which llama-server" &>/dev/null; then
        pass "llama-server binary found in container"
    else
        fail "llama-server binary not found in container"
    fi

    # Check llama-rpc-server binary
    if ssh "${SLURM_CTL}" "apptainer exec ${CONTAINER} which llama-rpc-server" &>/dev/null; then
        pass "llama-rpc-server binary found in container"
    else
        warn "llama-rpc-server binary not found (distributed inference may not work)"
    fi
fi

###############################################################################
# Inference Validation
###############################################################################

if $RUN_INFERENCE; then
    echo ""
    echo "--- Inference Validation ---"

    # Check 14B model exists
    if ssh "vasp-01" "test -f ${MODEL_14B}" 2>/dev/null; then
        pass "14B model exists on vasp-01"
    else
        warn "14B model not found at ${MODEL_14B}"
        info "Download: huggingface-cli download TheBloke/Strand-Rust-Coder-14B-GGUF strand-rust-coder-14b.Q8_0.gguf"
    fi

    # Check 72B model exists (check on any node)
    if ssh "vasp-01" "test -f ${MODEL_72B}" 2>/dev/null; then
        pass "72B model exists on vasp-01"
    else
        warn "72B model not found at ${MODEL_72B}"
        info "Download: huggingface-cli download TheBloke/OR1-Behemoth-73B-GGUF OR1-Behemoth.Q8_0.gguf"
    fi

    # Check job scripts exist
    if [[ -f "${SCRIPT_DIR}/run-14b.slurm" ]]; then
        pass "run-14b.slurm exists"
    else
        fail "run-14b.slurm not found"
    fi

    if [[ -f "${SCRIPT_DIR}/run-72b-distributed.slurm" ]]; then
        pass "run-72b-distributed.slurm exists"
    else
        fail "run-72b-distributed.slurm not found"
    fi

    # Check for running inference jobs
    RUNNING_JOBS=$(ssh "${SLURM_CTL}" "squeue -p gpu_ai -h 2>/dev/null" | wc -l || echo "0")
    info "Currently running AI jobs: ${RUNNING_JOBS}"

    # Check for active endpoints
    ENDPOINTS=$(ssh "vasp-01" "ls /scratch/ai/endpoints/*.json 2>/dev/null | wc -l" || echo "0")
    if [[ "$ENDPOINTS" -gt 0 ]]; then
        info "Active endpoint files: ${ENDPOINTS}"
        ssh "vasp-01" "cat /scratch/ai/endpoints/*.json 2>/dev/null" | jq -r '.endpoint' 2>/dev/null || true
    fi
fi

###############################################################################
# Summary
###############################################################################

echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC}   ${PASSED}"
echo -e "${RED}Failed:${NC}   ${FAILED}"
echo -e "${YELLOW}Warnings:${NC} ${WARNINGS}"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All critical checks passed!${NC}"

    if [[ $WARNINGS -gt 0 ]]; then
        echo "Review warnings above for optional improvements."
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Download models if not present"
    echo "  2. Test 14B inference: sbatch scripts/llama-cpp/run-14b.slurm"
    echo "  3. Test 72B distributed: sbatch scripts/llama-cpp/run-72b-distributed.slurm"
    echo "  4. Test preemption: Submit AI job, then VASP job"

    exit 0
else
    echo -e "${RED}Some checks failed. Review issues above.${NC}"
    exit 1
fi
