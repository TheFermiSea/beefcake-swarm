#!/bin/bash
###############################################################################
# Setup SLURM for AI Inference with Preemption
#
# This script configures SLURM to run AI inference jobs with preemptible
# scheduling. VASP jobs have priority and will preempt AI jobs.
#
# Usage:
#   ./setup-slurm-ai.sh          # Dry-run (show what would be done)
#   ./setup-slurm-ai.sh --apply  # Apply changes
#
# What this does:
#   1. Adds gpu_ai partition (preemptible)
#   2. Creates ai_opportunistic QoS with 30s grace time
#   3. Creates /scratch/ai directories on all nodes
#   4. Updates SLURM configuration
###############################################################################

set -euo pipefail

DRY_RUN=true
SLURM_CTL="slurm-ctl"
SLURM_CONF="/etc/slurm/slurm.conf"

# Node configuration - adjust if your nodes have different names
# These should match the NodeName entries in your slurm.conf
# Default: vasp-01, vasp-02, vasp-03 (per CLAUDE.md)
GPU_NODES="${GPU_NODES:-vasp-01 vasp-02 vasp-03}"
GPU_NODE_RANGE="${GPU_NODE_RANGE:-vasp-[01-03]}"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --apply)
            DRY_RUN=false
            ;;
        --help|-h)
            echo "Usage: $0 [--apply]"
            echo ""
            echo "Options:"
            echo "  --apply    Apply changes (default is dry-run)"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "SLURM AI Inference Setup"
echo "Mode: $(if $DRY_RUN; then echo 'DRY-RUN'; else echo 'APPLY'; fi)"
echo "=========================================="

# Function to run command or show it
# NOTE: eval is intentional here (SC2294) - commands contain nested quotes
# that require shell interpretation. All inputs are hardcoded, not user-supplied.
# shellcheck disable=SC2294
run_cmd() {
    if $DRY_RUN; then
        echo "[DRY-RUN] Would run: $*"
    else
        echo "[RUNNING] $*"
        eval "$@"
    fi
}

# Function to run command on slurm-ctl
run_on_ctl() {
    if $DRY_RUN; then
        echo "[DRY-RUN] Would run on ${SLURM_CTL}: $*"
    else
        echo "[RUNNING on ${SLURM_CTL}] $*"
        ssh "${SLURM_CTL}" "$@"
    fi
}

echo ""
echo "Step 1: Check current SLURM configuration"
echo "------------------------------------------"

# Check if gpu_ai partition already exists
if ssh "${SLURM_CTL}" "grep -q 'PartitionName=gpu_ai' ${SLURM_CONF} 2>/dev/null"; then
    echo "  gpu_ai partition already exists in slurm.conf"
    GPU_AI_EXISTS=true
else
    echo "  gpu_ai partition not found - will add"
    GPU_AI_EXISTS=false
fi

echo ""
echo "Step 2: Create /scratch/ai directories"
echo "---------------------------------------"
echo "  Target nodes: ${GPU_NODES}"

# Convert space-separated string to array
read -ra NODE_ARRAY <<< "${GPU_NODES}"

for node in "${NODE_ARRAY[@]}"; do
    echo "  Creating directories on ${node}..."
    run_cmd "ssh ${node} 'mkdir -p /scratch/ai/{models,logs,endpoints} && chmod 775 /scratch/ai /scratch/ai/*'"
done

echo ""
echo "Step 3: Configure global preemption settings"
echo "---------------------------------------------"

# Check and fix PreemptMode (must be REQUEUE, not CANCEL)
if ssh "${SLURM_CTL}" "grep -q 'PreemptMode=CANCEL' ${SLURM_CONF} 2>/dev/null"; then
    echo "  Changing PreemptMode from CANCEL to REQUEUE..."
    run_on_ctl "sed -i 's/PreemptMode=CANCEL/PreemptMode=REQUEUE/' ${SLURM_CONF}"
elif ssh "${SLURM_CTL}" "grep -q 'PreemptMode=REQUEUE' ${SLURM_CONF} 2>/dev/null"; then
    echo "  PreemptMode=REQUEUE already configured"
else
    echo "  Adding PreemptMode=REQUEUE..."
    run_on_ctl "sed -i '/^SlurmctldParameters/a PreemptMode=REQUEUE' ${SLURM_CONF}"
fi

# Check and fix PreemptType (must be preempt/qos for QoS-based preemption)
if ssh "${SLURM_CTL}" "grep -q 'PreemptType=preempt/qos' ${SLURM_CONF} 2>/dev/null"; then
    echo "  PreemptType=preempt/qos already configured"
elif ssh "${SLURM_CTL}" "grep -q 'PreemptType=preempt/partition_prio' ${SLURM_CONF} 2>/dev/null"; then
    echo "  Changing PreemptType from partition_prio to qos..."
    run_on_ctl "sed -i 's/PreemptType=preempt\\/partition_prio/PreemptType=preempt\\/qos/' ${SLURM_CONF}"
else
    echo "  Adding PreemptType=preempt/qos..."
    run_on_ctl "sed -i '/^PreemptMode/a PreemptType=preempt/qos' ${SLURM_CONF}"
fi

echo ""
echo "Step 4: Add GPU partitions to SLURM configuration"
echo "--------------------------------------------------"

# Check for gpu_vasp partition
if ssh "${SLURM_CTL}" "grep -q 'PartitionName=gpu_vasp' ${SLURM_CONF} 2>/dev/null"; then
    echo "  gpu_vasp partition already exists"
    GPU_VASP_EXISTS=true
else
    echo "  gpu_vasp partition not found - will add"
    GPU_VASP_EXISTS=false
fi

if ! $GPU_AI_EXISTS || ! $GPU_VASP_EXISTS; then
    # SLURM partition configuration to add
    PARTITION_CONFIG=""

    if ! $GPU_VASP_EXISTS; then
        PARTITION_CONFIG+="
# --- GPU VASP PARTITION (High Priority) ---
# VASP production jobs - can preempt AI inference
PartitionName=gpu_vasp Nodes=${GPU_NODE_RANGE} Default=NO MaxTime=7-00:00:00 State=UP PriorityJobFactor=1000 PriorityTier=2 PreemptMode=REQUEUE"
    fi

    if ! $GPU_AI_EXISTS; then
        PARTITION_CONFIG+="
# --- GPU AI PARTITION (Preemptible) ---
# AI inference jobs - yield to VASP with 30s grace time
PartitionName=gpu_ai Nodes=${GPU_NODE_RANGE} Default=NO MaxTime=24:00:00 State=UP PriorityJobFactor=1 PriorityTier=1 PreemptMode=REQUEUE GraceTime=30"
    fi

    echo "  Adding partition configuration..."
    if $DRY_RUN; then
        echo "[DRY-RUN] Would append to ${SLURM_CONF}:"
        echo "${PARTITION_CONFIG}"
    else
        ssh "${SLURM_CTL}" "echo '${PARTITION_CONFIG}' >> ${SLURM_CONF}"
    fi
else
    echo "  Skipping - both gpu_vasp and gpu_ai partitions already exist"
fi

echo ""
echo "Step 5: Configure QoS for preemption hierarchy"
echo "-----------------------------------------------"

# Check if QoS already exists
if ssh "${SLURM_CTL}" "sacctmgr show qos ai_opportunistic -n 2>/dev/null | grep -q ai_opportunistic"; then
    echo "  ai_opportunistic QoS already exists"
else
    echo "  Creating ai_opportunistic QoS..."
    run_on_ctl "sacctmgr -i add qos ai_opportunistic Priority=10 GraceTime=00:00:30 Description='Preemptible AI inference'"
fi

# Ensure VASP QoS can preempt AI jobs
if ssh "${SLURM_CTL}" "sacctmgr show qos vasp_priority -n 2>/dev/null | grep -q vasp_priority"; then
    echo "  vasp_priority QoS already exists"
else
    echo "  Creating vasp_priority QoS with preemption rights..."
    run_on_ctl "sacctmgr -i add qos vasp_priority Priority=1000 Preempt=ai_opportunistic Description='High-priority VASP jobs that can preempt AI'"
fi

echo ""
echo "Step 6: Reload SLURM configuration"
echo "-----------------------------------"

run_on_ctl "scontrol reconfigure"

echo ""
echo "Step 7: Verify configuration"
echo "-----------------------------"

echo "  Partitions:"
ssh "${SLURM_CTL}" "sinfo -o '%P %a %l %D %N' 2>/dev/null" | head -10

echo ""
echo "  QoS (should show vasp_priority and ai_opportunistic):"
ssh "${SLURM_CTL}" "sacctmgr show qos format=Name,Priority,Preempt,GraceTime -n 2>/dev/null" | grep -E "(vasp_priority|ai_opportunistic)" || echo "    (check with 'sacctmgr show qos')"

echo ""
echo "  Preemption settings:"
ssh "${SLURM_CTL}" "grep -E '^Preempt(Type|Mode)' ${SLURM_CONF} 2>/dev/null" || echo "    (check slurm.conf manually)"

echo ""
echo "=========================================="
if $DRY_RUN; then
    echo "DRY-RUN complete. Run with --apply to make changes."
else
    echo "Setup complete!"
    echo ""
    echo "To submit an AI job:"
    echo "  sbatch --partition=gpu_ai --qos=ai_opportunistic run-14b.slurm"
    echo ""
    echo "To verify preemption works:"
    echo "  1. Submit an AI job to gpu_ai"
    echo "  2. Submit a VASP job to critical or normal"
    echo "  3. AI job should be requeued with 30s grace time"
fi
echo "=========================================="
