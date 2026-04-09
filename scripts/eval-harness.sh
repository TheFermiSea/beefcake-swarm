#!/usr/bin/env bash
# Declarative evaluation harness for Beefcake Swarm
# Inspired by ForgeCode benchmarks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${REPO_ROOT}/benchmarks"
EVAL_LOGS="${REPO_ROOT}/logs/eval"

mkdir -p "$EVAL_LOGS"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark_spec.json>"
    echo "Example: $0 benchmarks/sample-eval.json"
    exit 1
fi

SPEC_FILE="$1"

if [[ ! -f "$SPEC_FILE" ]]; then
    echo "Error: Spec file not found: $SPEC_FILE"
    exit 1
fi

echo "Starting evaluation harness using $SPEC_FILE"

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required"
    exit 1
fi

# Parse tasks from JSON
TASKS_COUNT=$(jq '.tasks | length' "$SPEC_FILE")

SUCCESS_COUNT=0
FAILURE_COUNT=0

for ((i=0; i<TASKS_COUNT; i++)); do
    TASK_ID=$(jq -r ".tasks[$i].id" "$SPEC_FILE")
    TASK_TITLE=$(jq -r ".tasks[$i].title" "$SPEC_FILE")
    TASK_DESC=$(jq -r ".tasks[$i].description" "$SPEC_FILE")
    TIMEOUT=$(jq -r ".tasks[$i].timeout // 600" "$SPEC_FILE")
    VALIDATION_CMD=$(jq -r ".tasks[$i].validation_command // \"\"" "$SPEC_FILE")

    echo "============================================================"
    echo "Running Task [$TASK_ID]: $TASK_TITLE"
    
    RUN_LOG="$EVAL_LOGS/eval-${TASK_ID}-$(date +%Y%m%d-%H%M%S).log"
    
    # Build a rich objective packet
    OBJECTIVE_FILE="$EVAL_LOGS/objective-${TASK_ID}.txt"
    cat <<EOF > "$OBJECTIVE_FILE"
Issue ID: $TASK_ID
Title: $TASK_TITLE

=== Description ===
$TASK_DESC
EOF

    echo "  -> Log: $RUN_LOG"
    
    run_start=$(date +%s)
    set +e
    
    # Run the swarm
    # We use timeout to enforce the time limit
    timeout "$TIMEOUT" bash "$REPO_ROOT/scripts/run-swarm.sh" \
        --issue "$TASK_ID" \
        --objective "$(cat "$OBJECTIVE_FILE")" \
        > "$RUN_LOG" 2>&1
    
    EXIT_CODE=$?
    set -e
    run_end=$(date +%s)
    elapsed=$((run_end - run_start))
    
    if [[ $EXIT_CODE -eq 124 ]]; then
        echo "  [FAIL] Task $TASK_ID timed out after ${TIMEOUT}s"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        continue
    elif [[ $EXIT_CODE -ne 0 ]]; then
        echo "  [FAIL] Task $TASK_ID failed with exit code $EXIT_CODE (${elapsed}s)"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        continue
    fi
    
    # If there's a validation command, run it
    if [[ -n "$VALIDATION_CMD" && "$VALIDATION_CMD" != "null" ]]; then
        echo "  -> Running validation: $VALIDATION_CMD"
        set +e
        eval "$VALIDATION_CMD" >> "$RUN_LOG" 2>&1
        VAL_EXIT=$?
        set -e
        
        if [[ $VAL_EXIT -eq 0 ]]; then
            echo "  [PASS] Task $TASK_ID succeeded and passed validation (${elapsed}s)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  [FAIL] Task $TASK_ID succeeded but FAILED validation (${elapsed}s)"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
        fi
    else
        echo "  [PASS] Task $TASK_ID succeeded (no validation command) (${elapsed}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

echo "============================================================"
echo "Evaluation Complete!"
echo "Passed: $SUCCESS_COUNT"
echo "Failed: $FAILURE_COUNT"
echo "Total: $TASKS_COUNT"
