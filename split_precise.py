import re
import os

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

def find_item_bounds(name):
    start = -1
    for i, line in enumerate(lines):
        if re.match(r'^((pub|pub\(crate\)) )?(struct|enum|fn|const|mod|impl) ' + name + r'\b', line.strip()) or re.match(r'^impl( [A-Za-z0-9<>_]+ for)? ' + name + r'\b', line.strip()):
            start = i
            while start > 0 and (lines[start-1].strip().startswith('#[') or lines[start-1].strip().startswith('///') or lines[start-1].strip().startswith('//') or lines[start-1].strip() == ''):
                start -= 1
            # Find end by counting braces
            brace_count = 0
            started = False
            for j in range(i, len(lines)):
                brace_count += lines[j].count('{')
                brace_count -= lines[j].count('}')
                if '{' in lines[j]:
                    started = True
                if started and brace_count == 0:
                    return (start, j)
            if not started and lines[i].strip().endswith(';'):
                return (start, i)
    return None

items_map = {
    'types': [
        'ArtifactAction', 'ArtifactRecord', 'ARTIFACT_SCHEMA_VERSION', 'RouteDecision',
        'VerifierSnapshot', 'GateSnapshot', 'EvaluatorSnapshot', 'RetryAction',
        'RetryRationale', 'ExecutionArtifact', 'IterationMetrics', 'SessionMetrics',
        'HarnessComponentTrace', 'ValidationMetric', 'IterationBuilder', 'FailureLedgerEntry',
        'SloStatus', 'SloMeasurement', 'SloTargets', 'SloReport', 'cost_rates'
    ],
    'aggregation': [
        'MetricsCollector', 'artifact_churn_score', 'AggregateAnalytics', 'LoopMetrics',
        'TelemetryReader', 'measure', 'CostTracker', 'prune_task_prompt'
    ],
    'export': [
        'write_session_metrics', 'append_telemetry', 'append_experiment_tsv',
        'append_failure_ledger', 'write_execution_artifacts', 'prune_artifact_sessions'
    ],
    'events': [
        'SwarmEvent', 'SwarmEventPayload', 'SwarmEventEmitter'
    ]
}

# we need to get ALL impls. A struct can have multiple impls!
def find_all_item_bounds(name):
    bounds = []
    skip_until = -1
    for i, line in enumerate(lines):
        if i < skip_until: continue
        if re.match(r'^((pub|pub\(crate\)) )?(struct|enum|fn|const|mod|impl) ' + name + r'\b', line.strip()) or re.match(r'^impl( [A-Za-z0-9<>_]+ for)? ' + name + r'\b', line.strip()) or re.match(r'^impl .* for ' + name + r'\b', line.strip()):
            start = i
            while start > 0 and (lines[start-1].strip().startswith('#[') or lines[start-1].strip().startswith('///') or lines[start-1].strip() == ''):
                start -= 1
            brace_count = 0
            started = False
            for j in range(i, len(lines)):
                brace_count += lines[j].count('{')
                brace_count -= lines[j].count('}')
                if '{' in lines[j]:
                    started = True
                if started and brace_count == 0:
                    bounds.append((start, j))
                    skip_until = j + 1
                    break
            if not started and lines[i].strip().endswith(';'):
                bounds.append((start, i))
                skip_until = i + 1
    return bounds

import collections
files = collections.defaultdict(list)

for file_key, names in items_map.items():
    for name in names:
        bounds_list = find_all_item_bounds(name)
        files[file_key].extend(bounds_list)

# Now what about tests?
tests_start = 1646
test_blocks = []
current_block = []
for i in range(tests_start, len(lines)):
    line = lines[i]
    if line.strip() == '#[test]' or line.strip() == '#[tokio::test]':
        if current_block: test_blocks.append(current_block)
        current_block = [line]
    else:
        if current_block: current_block.append(line)
if current_block: test_blocks.append(current_block)

def get_test_category(block):
    text = "".join(block)
    if 'metrics_collector' in text or 'finalize_flushes' in text or 'telemetry_reader' in text or 'artifact_churn_score' in text or 'record_artifact' in text or 'test_slo_' in text or 'cost_tracker' in text:
        return 'aggregation'
    if 'append_telemetry_jsonl' in text or 'write_execution_artifacts' in text or 'write_session_metrics' in text or 'artifact_retention' in text:
        return 'export'
    if 'execution_artifact' in text or 'retry_action' in text:
        return 'types'
    if 'swarm_event' in text or 'event_emitter' in text or 'is_critical' in text:
        return 'events'
    return 'init'

test_files = collections.defaultdict(list)
for b in test_blocks:
    cat = get_test_category(b)
    test_files[cat].extend(b)

header = """use super::*;
use std::path::Path;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

"""

for fkey in ['types', 'aggregation', 'export', 'events', 'init']:
    with open(f'crates/swarm-agents/src/telemetry/{fkey}.rs', 'w') as f:
        f.write(header)
        # Sort bounds
        b_list = sorted(files[fkey], key=lambda x: x[0])
        for s, e in b_list:
            f.write("".join(lines[s:e+1]) + "\n")

        if test_files[fkey]:
            f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use tempfile::tempdir;\n    use std::fs;\n")
            f.write("".join(test_files[fkey]))
            f.write("\n}\n")
