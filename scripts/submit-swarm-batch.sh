#!/usr/bin/env bash
# Submit a batch of swarm-ready beads issues to SLURM.
#
# Usage:
#   ./scripts/submit-swarm-batch.sh                    # submit all swarm-ready issues
#   ./scripts/submit-swarm-batch.sh xv9 a3y 4qz       # submit specific issues
#   DRY_RUN=1 ./scripts/submit-swarm-batch.sh          # preview without submitting
#
# Each issue runs as a separate SLURM job using --issue flag for deterministic targeting.
# Jobs run sequentially by default (each depends on the previous via --dependency=afterany).
# Set PARALLEL=1 to submit all jobs independently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run-swarm-sandbox.slurm"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"

# Collect issue IDs
if [ $# -gt 0 ]; then
    ISSUES=("$@")
else
    # Auto-discover swarm-ready issues via beads label
    mapfile -t ISSUES < <(
        bd list --status=open --label=swarm-ready 2>/dev/null \
            | python3 -c "import sys,json; [print(i['id'].split('-')[-1]) for i in json.load(sys.stdin)]" 2>/dev/null \
            || echo ""
    )
    if [ ${#ISSUES[@]} -eq 0 ] || [ -z "${ISSUES[0]}" ]; then
        echo "No swarm-ready issues found. Label issues with 'swarm-ready' first."
        exit 1
    fi
fi

echo "=== Swarm Batch Submission ==="
echo "Issues: ${ISSUES[*]}"
echo "Mode: $([ "$PARALLEL" = "1" ] && echo "parallel" || echo "sequential")"
echo "Dry run: $([ "$DRY_RUN" = "1" ] && echo "yes" || echo "no")"
echo ""

PREV_JOB=""
SUBMITTED=()

for issue_id in "${ISSUES[@]}"; do
    full_id="beefcake-swarm-${issue_id}"

    # Fetch title for logging
    title=$(bd show "$full_id" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)[0].get('title','?'))" 2>/dev/null || echo "?")

    echo "[$full_id] $title"

    # Build sbatch command with --issue override
    SBATCH_ARGS=(
        --job-name="swarm-${issue_id}"
        --export="ALL,SWARM_ISSUE_ID=${full_id}"
    )

    # Sequential mode: chain jobs
    if [ "$PARALLEL" != "1" ] && [ -n "$PREV_JOB" ]; then
        SBATCH_ARGS+=(--dependency="afterany:${PREV_JOB}")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] sbatch ${SBATCH_ARGS[*]} $SLURM_SCRIPT"
    else
        JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  Submitted: job $JOB_ID"
        PREV_JOB="$JOB_ID"
        SUBMITTED+=("$JOB_ID")

        # Mark issue as in_progress
        bd update "$full_id" --status=in_progress 2>/dev/null || true
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${#SUBMITTED[@]} jobs: ${SUBMITTED[*]}"
    echo "Monitor: squeue -u brian | grep swarm"
    echo "Logs: /cluster/shared/ai/logs/swarm-orch-*.log"
fi
