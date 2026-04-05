import re

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

tests_start = 1646
test_lines = lines[tests_start:]

# Put test blocks in the files that define the things they test.
# Or put them in a separate test mod. I'll append `#[cfg(test)] mod tests { super::*; ... }` to each file depending on what the tests are for.
# Test 0-3, 5, 10: aggregation
# Test 4, 25-26: export
# Test 6-9: aggregation (artifact_churn_score)
# Test 11-17: types (slo) or aggregation? TelemetryReader is aggregation. slo is in aggregation/types.
# Test 18-24: types (execution artifact)
# Test 27-29: events

blocks = []
current_block = []

for line in test_lines:
    if line.strip() == '#[test]' or line.strip() == '#[tokio::test]':
        if current_block:
            blocks.append(current_block)
        current_block = [line]
    else:
        if current_block:
            current_block.append(line)
if current_block:
    blocks.append(current_block)

def get_test_category(block):
    text = "".join(block)
    if 'metrics_collector' in text or 'finalize_flushes' in text or 'telemetry_reader' in text or 'artifact_churn_score' in text or 'record_artifact' in text or 'test_slo_' in text:
        return 'aggregation.rs'
    if 'append_telemetry_jsonl' in text or 'write_execution_artifacts' in text or 'write_session_metrics' in text or 'artifact_retention' in text:
        return 'export.rs'
    if 'execution_artifact' in text or 'retry_action' in text:
        return 'types.rs'
    if 'swarm_event' in text or 'event_emitter' in text or 'is_critical' in text:
        return 'events.rs'
    return 'init.rs'

files_tests = {
    'aggregation.rs': [],
    'export.rs': [],
    'types.rs': [],
    'events.rs': [],
    'init.rs': []
}

for b in blocks:
    cat = get_test_category(b)
    files_tests[cat].extend(b)

for fname, lines_list in files_tests.items():
    if lines_list:
        with open(f'crates/swarm-agents/src/telemetry/{fname}', 'a') as f:
            f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use tempfile::tempdir;\n    use std::fs;\n")
            f.write("".join(lines_list))
            f.write("\n}\n")
