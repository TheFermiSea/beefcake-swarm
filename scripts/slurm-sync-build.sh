#!/usr/bin/env bash
# slurm-sync-build.sh — Sync ai-proxy's repo to NFS and rebuild the release binary.
#
# ai-proxy has its own /home (not NFS-mounted). The compute nodes (vasp-01/02/03)
# mount /home from slurm-ctl (10.0.0.5) via NFS. This script:
#
#   1. Pushes the current ai-proxy branch to the NFS repo on slurm-ctl
#   2. Optionally rebuilds the release binary via a SLURM job
#
# Usage:
#   ./scripts/slurm-sync-build.sh              # Sync only
#   ./scripts/slurm-sync-build.sh --build      # Sync + rebuild binary
#   ./scripts/slurm-sync-build.sh --build-only # Rebuild without sync (assumes already synced)
#   ./scripts/slurm-sync-build.sh --status     # Show NFS repo and binary status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_CTL_HOST="${SLURM_CTL_HOST:-root@10.0.0.5}"
NFS_REPO="/home/brian/code/beefcake-swarm"
NFS_BIN="/cluster/shared/ai/bin/swarm-agents"
CARGO_TARGET="/cluster/shared/cargo-cache/beefcake-target"
SLURM_PARTITION="${SLURM_PARTITION:-gpu_ai}"
SLURM_QOS="${SLURM_QOS:-ai_opportunistic}"
BUILD_NODE="${BUILD_NODE:-vasp-01}"

DO_SYNC=1
DO_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)      DO_BUILD=1; shift ;;
        --build-only) DO_SYNC=0; DO_BUILD=1; shift ;;
        --status)
            echo "=== NFS Repo Status ==="
            ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
                "git -C '$NFS_REPO' log --oneline -3 && echo '---' && git -C '$NFS_REPO' status --short 2>/dev/null | head -10"
            echo ""
            echo "=== Local (ai-proxy) Repo Status ==="
            git -C "$REPO_ROOT" log --oneline -3
            echo ""
            echo "=== Release Binary ==="
            ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
                "ls -lh '$NFS_BIN' 2>/dev/null || echo 'NOT FOUND'"
            echo ""
            echo "=== SLURM Nodes ==="
            sinfo -N -p gpu_ai --noheader 2>/dev/null || echo "sinfo unavailable"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $(basename "$0") [--build|--build-only|--status|--help]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Sync ---
if [[ "$DO_SYNC" -eq 1 ]]; then
    log "Syncing ai-proxy repo to NFS..."

    # Check NFS repo exists
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SLURM_CTL_HOST" \
        "test -d '${NFS_REPO}/.git'" 2>/dev/null; then
        log "NFS repo not found — cloning..."
        ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
            "su - brian -c 'git clone git@github.com:TheFermiSea/beefcake-swarm.git ${NFS_REPO}'" 2>&1
    fi

    # Add NFS repo as a git remote (if not already)
    REMOTE_NAME="nfs-slurm"
    if ! git -C "$REPO_ROOT" remote get-url "$REMOTE_NAME" &>/dev/null; then
        git -C "$REPO_ROOT" remote add "$REMOTE_NAME" "ssh://root@10.0.0.5${NFS_REPO}"
        log "Added git remote: $REMOTE_NAME"
    fi

    # Push current branch
    CURRENT_BRANCH=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo "main")
    log "Pushing $CURRENT_BRANCH to NFS repo..."
    if git -C "$REPO_ROOT" push "$REMOTE_NAME" "${CURRENT_BRANCH}:${CURRENT_BRANCH}" --force 2>&1 | sed 's/^/  /'; then
        log "Push succeeded"
    else
        log "WARNING: Push failed — trying rsync fallback"
        rsync -a --delete \
            --exclude='.git' --exclude='target/' --exclude='logs/' --exclude='.swarm/' \
            "$REPO_ROOT/" "root@10.0.0.5:${NFS_REPO}/" 2>&1 | tail -5 | sed 's/^/  /'
    fi

    # Checkout the pushed branch on NFS
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$SLURM_CTL_HOST" \
        "git -C '$NFS_REPO' checkout '${CURRENT_BRANCH}' --force 2>&1" | sed 's/^/  /'

    # Show sync result
    log "NFS repo HEAD:"
    ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
        "git -C '$NFS_REPO' log --oneline -1" | sed 's/^/  /'
fi

# --- Build ---
if [[ "$DO_BUILD" -eq 1 ]]; then
    log "Submitting release build to SLURM on $BUILD_NODE..."

    NFS_LOG_DIR="${NFS_REPO}/logs/slurm-agents"
    ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
        "su - brian -c 'mkdir -p $NFS_LOG_DIR'" 2>/dev/null || true

    BUILD_OUTPUT=$(sbatch \
        --job-name="swarm-build" \
        --nodelist="$BUILD_NODE" \
        --partition="$SLURM_PARTITION" \
        --qos="$SLURM_QOS" \
        --time="00:30:00" \
        --cpus-per-task=8 \
        --mem=32G \
        --uid=brian \
        --gid=hpc \
        --output="${NFS_LOG_DIR}/build-%j.log" \
        --error="${NFS_LOG_DIR}/build-%j.err" \
        --wrap="bash -c '
            source \$HOME/.cargo/env 2>/dev/null || true
            export CARGO_TARGET_DIR=${CARGO_TARGET}
            cd ${NFS_REPO}
            echo \"Building swarm-agents (release)...\"
            cargo build -p swarm-agents --release 2>&1
            if [[ -f \$CARGO_TARGET_DIR/release/swarm-agents ]]; then
                cp \$CARGO_TARGET_DIR/release/swarm-agents ${NFS_BIN}
                echo \"Binary installed: ${NFS_BIN} (\$(ls -lh ${NFS_BIN} | awk \"{print \\\$5}\"))\"
            else
                echo \"ERROR: Build produced no binary\"
                exit 1
            fi
        '" 2>&1)

    JOB_ID=$(echo "$BUILD_OUTPUT" | grep -oP '\d+' | head -1)
    if [[ -z "$JOB_ID" ]]; then
        log "ERROR: Failed to submit build job: $BUILD_OUTPUT"
        exit 1
    fi

    log "Build job $JOB_ID submitted. Waiting..."

    # Poll for completion
    while true; do
        STATE=$(squeue -j "$JOB_ID" --noheader --format="%T" 2>/dev/null | tr -d ' ' || echo "")
        if [[ -z "$STATE" ]]; then
            STATE=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$SLURM_CTL_HOST" \
                "/usr/local/bin/sacct -j $JOB_ID --noheader --format=State --parsable2" 2>/dev/null \
                | head -1 | tr -d ' ' || echo "COMPLETED")
            break
        fi
        if [[ "$STATE" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT)$ ]]; then break; fi
        printf "."
        sleep 5
    done
    echo ""

    if [[ "$STATE" == "COMPLETED" ]]; then
        log "Build succeeded"
        ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" "ls -lh '$NFS_BIN'" | sed 's/^/  /'
    else
        log "Build FAILED (state=$STATE). Log:"
        ssh -o ConnectTimeout=10 "$SLURM_CTL_HOST" \
            "cat ${NFS_LOG_DIR}/build-${JOB_ID}.log 2>/dev/null | tail -30" | sed 's/^/  /'
        exit 1
    fi
fi

log "Done."
