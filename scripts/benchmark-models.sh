#!/usr/bin/env bash
# benchmark-models.sh — Run standardized dogfood issues across different model configs.
#
# Usage:
#   ./scripts/benchmark-models.sh                    # Run all configs
#   ./scripts/benchmark-models.sh --config opus-27b  # Run specific config
#   ./scripts/benchmark-models.sh --dry-run          # Show what would run
#
# Output: logs/benchmark/<config>/<issue>/<run>.log
#         logs/benchmark/summary.jsonl
#
# The benchmark runs the same issues with different model configurations,
# tracking time, iterations, tool calls, and success rate for comparison.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/logs/benchmark"
SUMMARY_FILE="${BENCHMARK_DIR}/summary.jsonl"
ITERATIONS_PER_CONFIG="${BENCHMARK_ITERATIONS:-1}"
COOLDOWN=30

# Standard test issues — mix of simple and complex
# These should be pre-created beads issues with known-good descriptions
BENCHMARK_ISSUES=(
    # Simple mechanical (should succeed in 1 iteration)
    "beefcake-dbsa"   # Replace .unwrap() with .expect()
    "beefcake-irit"   # Replace 2 .unwrap() with ?
    "beefcake-j5vc"   # Replace 2 .unwrap() with ?
    "beefcake-lfvg"   # Replace .unwrap() with ?
    # Complex (may need cloud/multiple iterations)
    "beefcake-pl2l"   # Add omission guard feature
)

# Model configurations to benchmark
declare -A CONFIGS
CONFIGS[opus-distilled-27b]="SWARM_FAST_MODEL=Qwen3.5-27B-Opus-Distilled SWARM_FAST_URL=http://vasp-03:8081/v1"
CONFIGS[original-27b-distilled]="SWARM_FAST_MODEL=Qwen3.5-27B-Distilled SWARM_FAST_URL=http://vasp-03:8081/v1"
CONFIGS[coder-122b]="SWARM_FAST_MODEL=Qwen3.5-122B-A10B SWARM_FAST_URL=http://vasp-01:8081/v1"
CONFIGS[cloud-only]="SWARM_CLOUD_ONLY=true"

CONFIG_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --iterations) ITERATIONS_PER_CONFIG="$2"; shift 2 ;;
        --issues) IFS=',' read -ra BENCHMARK_ISSUES <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$BENCHMARK_DIR"

log() { echo "[benchmark] $(date '+%H:%M:%S') $*"; }

log "Benchmark configuration:"
log "  Issues: ${BENCHMARK_ISSUES[*]}"
log "  Iterations per config: $ITERATIONS_PER_CONFIG"
log "  Configs: ${!CONFIGS[*]}"
if [[ -n "$CONFIG_FILTER" ]]; then
    log "  Filter: $CONFIG_FILTER"
fi

if $DRY_RUN; then
    log "DRY RUN — would execute:"
    for config_name in "${!CONFIGS[@]}"; do
        [[ -n "$CONFIG_FILTER" && "$config_name" != "$CONFIG_FILTER" ]] && continue
        log "  Config: $config_name"
        log "    Env: ${CONFIGS[$config_name]}"
        for issue in "${BENCHMARK_ISSUES[@]}"; do
            for iter in $(seq 1 "$ITERATIONS_PER_CONFIG"); do
                log "    Run: $issue (iteration $iter)"
            done
        done
    done
    exit 0
fi

# Run benchmark
for config_name in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    [[ -n "$CONFIG_FILTER" && "$config_name" != "$CONFIG_FILTER" ]] && continue

    config_dir="${BENCHMARK_DIR}/${config_name}"
    mkdir -p "$config_dir"

    log "=== Config: $config_name ==="
    log "  Env: ${CONFIGS[$config_name]}"

    for iter in $(seq 1 "$ITERATIONS_PER_CONFIG"); do
        for issue in "${BENCHMARK_ISSUES[@]}"; do
            run_log="${config_dir}/${issue}-iter${iter}-$(date +%Y%m%d-%H%M%S).log"

            log "  Running: $issue (iter $iter) → $run_log"

            run_start=$(date +%s)
            set +e
            (
                cd "$REPO_ROOT"
                # Apply config-specific env vars
                eval "${CONFIGS[$config_name]}"
                export SWARM_FAST_MODEL SWARM_FAST_URL SWARM_CLOUD_ONLY 2>/dev/null || true
                bash scripts/run-swarm.sh --issue "$issue" --objective "$(bd show "$issue" --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); i=d.get("bd_stdout",d); i=i[0] if isinstance(i,list) else i; print(i.get("title","")+" "+i.get("description","")[:300])' 2>/dev/null || echo "Issue $issue")"
            ) > "$run_log" 2>&1
            exit_code=$?
            set -e
            run_end=$(date +%s)
            elapsed=$((run_end - run_start))

            # Extract metrics from log
            resolved=$(grep -c 'Issue resolved' "$run_log" 2>/dev/null || echo 0)
            iterations=$(grep -c 'Starting iteration' "$run_log" 2>/dev/null || echo 0)
            tool_calls=$(grep -c 'Tool call completed' "$run_log" 2>/dev/null || echo 0)
            short_circuit=$(grep -c 'short-circuit.*Accept' "$run_log" 2>/dev/null || echo 0)

            # Record to summary
            printf '{"config":"%s","issue":"%s","iteration":%d,"exit_code":%d,"elapsed_s":%d,"resolved":%d,"iterations":%d,"tool_calls":%d,"short_circuit":%d,"timestamp":"%s"}\n' \
                "$config_name" "$issue" "$iter" "$exit_code" "$elapsed" "$resolved" "$iterations" "$tool_calls" "$short_circuit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                >> "$SUMMARY_FILE"

            status="FAIL"
            [[ "$resolved" -gt 0 ]] && status="PASS"
            log "    Result: $status (${elapsed}s, ${iterations} iters, ${tool_calls} tools)"

            sleep "$COOLDOWN"
        done
    done
done

log "=== Benchmark complete ==="
log "Summary: $SUMMARY_FILE"

# Print summary table
if [[ -f "$SUMMARY_FILE" ]]; then
    log ""
    log "Results:"
    python3 -c "
import json, sys
from collections import defaultdict

results = defaultdict(list)
with open('$SUMMARY_FILE') as f:
    for line in f:
        r = json.loads(line)
        results[r['config']].append(r)

print(f'{'Config':<25s} {'Pass':>4s} {'Fail':>4s} {'Rate':>5s} {'Avg Time':>8s} {'Avg Tools':>9s}')
print('-' * 60)
for config in sorted(results):
    runs = results[config]
    passed = sum(1 for r in runs if r['resolved'] > 0)
    failed = len(runs) - passed
    rate = f'{100*passed/len(runs):.0f}%'
    avg_time = f'{sum(r[\"elapsed_s\"] for r in runs)/len(runs):.0f}s'
    avg_tools = f'{sum(r[\"tool_calls\"] for r in runs)/len(runs):.0f}'
    print(f'{config:<25s} {passed:>4d} {failed:>4d} {rate:>5s} {avg_time:>8s} {avg_tools:>9s}')
" 2>/dev/null || true
fi
