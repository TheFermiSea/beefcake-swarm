#!/usr/bin/env bash
# slurm-agent-dispatch.sh — Dispatch swarm worker agents across SLURM nodes.
#
# Instead of running all agents on ai-proxy, this script submits SLURM jobs
# that run swarm-agents workers on the compute nodes. Each node runs inference
# locally AND executes agent tool calls locally, avoiding network round-trips.
#
# Architecture (RepoProver pattern):
#   - Coordinator: ai-proxy (picks issues, delegates, merges)
#   - Workers: vasp-01/02/03 (run swarm-agents binary with --issue flag)
#
# Each node already has an inference server running on :8081. When a worker
# runs on vasp-01, it talks to Qwen3.5-27B at localhost:8081 — zero network
# overhead. Cross-tier calls still go over the network to the other nodes.
#
# Usage:
#   ./scripts/slurm-agent-dispatch.sh --issue beefcake-abc123
#   ./scripts/slurm-agent-dispatch.sh --issue-list "id1 id2 id3"
#   ./scripts/slurm-agent-dispatch.sh --parallel 3  # one per node
#   DRY_RUN=1 ./scripts/slurm-agent-dispatch.sh --issue-list "id1 id2"
#
# Node → Model mapping:
#   vasp-01 → Qwen3.5-27B (coder)
#   vasp-02 → Devstral-Small-2-24B (reasoning)
#   vasp-03 → GLM-4.7-Flash (scout/fast)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# SLURM configuration
SLURM_PARTITION="${SLURM_PARTITION:-gpu_ai}"
SLURM_QOS="${SLURM_QOS:-ai_opportunistic}"
SLURM_NODES="${SLURM_NODES:-vasp-01,vasp-02,vasp-03}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REPO_ROOT}/logs/slurm-agents}"
BEADS_BIN="${SWARM_BEADS_BIN:-${SCRIPT_DIR}/bd-safe.sh}"
DRY_RUN="${DRY_RUN:-0}"

# Parse args
ISSUE_LIST=""
PARALLEL=3
DISCOVER=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --issue ID           Single issue to dispatch
  --issue-list "IDs"   Space-separated list of issue IDs
  --parallel N         Max concurrent jobs (default: 3, one per node)
  --discover           Auto-fetch swarm-ready issues from beads
  --partition P        SLURM partition (default: gpu_ai)
  --time HH:MM:SS     SLURM time limit (default: 02:00:00)
  --help               Show this help

Environment:
  DRY_RUN=1           Preview sbatch commands without submitting
  SLURM_NODES         Comma-separated node list (default: vasp-01,vasp-02,vasp-03)
  SWARM_CLOUD_URL     Cloud proxy endpoint (default: http://10.0.0.5:8317/v1)
  SWARM_CLOUD_API_KEY Cloud API key (required for cloud manager mode)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --issue)      ISSUE_LIST="$2"; shift 2 ;;
        --issue-list) ISSUE_LIST="$2"; shift 2 ;;
        --parallel)   PARALLEL="$2"; shift 2 ;;
        --discover)   DISCOVER=1; shift ;;
        --partition)  SLURM_PARTITION="$2"; shift 2 ;;
        --time)       SLURM_TIME="$2"; shift 2 ;;
        --help|-h)    usage ;;
        *)            echo "Unknown arg: $1"; usage ;;
    esac
done

# Auto-discover swarm-ready issues if --discover and no issues given
if [[ -z "$ISSUE_LIST" && "$DISCOVER" == "1" ]]; then
    echo "[dispatch] Discovering swarm-ready issues..."
    ISSUE_LIST="$("$BEADS_BIN" list --status=open --label=swarm-ready 2>/dev/null \
        | python3 -c "import sys,json; [print(i['id']) for i in json.load(sys.stdin)]" 2>/dev/null \
        | head -n "$PARALLEL" \
        | tr '\n' ' ')" || true
    if [[ -z "${ISSUE_LIST// /}" ]]; then
        echo "[dispatch] No swarm-ready issues found. Label issues with 'swarm-ready' or pass --issue."
        exit 1
    fi
    echo "[dispatch] Discovered: $ISSUE_LIST"
fi

if [[ -z "$ISSUE_LIST" ]]; then
    echo "Error: No issues specified. Use --issue, --issue-list, or --discover."
    usage
fi

mkdir -p "$SLURM_LOG_DIR"

# Parse node list and issue list
IFS=' ' read -ra ISSUES <<< "$ISSUE_LIST"
IFS=',' read -ra NODES <<< "$SLURM_NODES"
NODE_COUNT=${#NODES[@]}

echo "=== SLURM Agent Dispatch ==="
echo "Issues:    ${ISSUES[*]}"
echo "Nodes:     ${NODES[*]}"
echo "Partition: $SLURM_PARTITION"
echo "Time:      $SLURM_TIME"
echo "Parallel:  $PARALLEL"
echo "Dry run:   $([ "$DRY_RUN" = "1" ] && echo "yes" || echo "no")"
echo ""

SUBMITTED=()

for i in "${!ISSUES[@]}"; do
    issue="${ISSUES[$i]}"
    node="${NODES[$((i % NODE_COUNT))]}"

    # Fetch title for objective context
    title=$("$BEADS_BIN" show "$issue" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0].get('title','') if isinstance(d,list) else d.get('title',''))" 2>/dev/null \
        || echo "$issue")
    objective_b64=$(printf "%s" "$title" | base64 | tr -d '\n')

    echo "[$issue] → $node ($title)"

    SBATCH_ARGS=(
        --job-name="swarm-agent-${issue##*-}"
        --nodelist="$node"
        --partition="$SLURM_PARTITION"
        --qos="$SLURM_QOS"
        --time="$SLURM_TIME"
        --output="${SLURM_LOG_DIR}/agent-%j-${issue##*-}.log"
        --error="${SLURM_LOG_DIR}/agent-%j-${issue##*-}.err"
        --export="ALL,ISSUE_ID=${issue},ISSUE_OBJECTIVE_B64=${objective_b64},DISPATCH_NODE=${node}"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] sbatch ${SBATCH_ARGS[*]} $SCRIPT_DIR/slurm-agent-worker.sh"
    else
        JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/slurm-agent-worker.sh" 2>&1 | grep -oP '\d+')
        echo "  Submitted: SLURM job $JOB_ID"
        SUBMITTED+=("$JOB_ID")

        # Mark issue as in_progress
        "$BEADS_BIN" update "$issue" --status=in_progress 2>/dev/null || true
    fi
done

echo ""
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${#SUBMITTED[@]} jobs: ${SUBMITTED[*]}"
    echo "Monitor:  squeue -u brian | grep swarm-agent"
    echo "Logs:     ${SLURM_LOG_DIR}/agent-*.log"
fi
