import re

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

def find_block(start_line_prefix, include_impls=True, stop_before=None):
    # simple block finder
    pass

# Instead of parsing in python, I'll use simple search to find line numbers
for i, line in enumerate(lines):
    if line.startswith('pub enum ArtifactAction'):
        print(f"ArtifactAction: {i}")
    if line.startswith('pub fn write_session_metrics'):
        print(f"write_session_metrics: {i}")
    if line.startswith('pub struct MetricsCollector'):
        print(f"MetricsCollector: {i}")
    if line.startswith('pub struct CostTracker'):
        print(f"CostTracker: {i}")
    if line.startswith('pub fn prune_task_prompt'):
        print(f"prune_task_prompt: {i}")
    if line.startswith('pub struct SwarmEvent'):
        print(f"SwarmEvent: {i}")
