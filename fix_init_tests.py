# The tests in init.rs are actually testing prune_task_prompt and append_experiment_tsv, which were moved.
import os

with open('crates/swarm-agents/src/telemetry/init.rs', 'r') as f:
    text = f.read()

# We need to remove the whole `mod tests { ... }` block from init.rs and types.rs and place them correctly, or just use `use crate::telemetry::*;` inside tests.
# Changing `use super::*;` to `use crate::telemetry::*;` in all test mods.

for fname in ['aggregation.rs', 'export.rs', 'types.rs', 'events.rs', 'init.rs']:
    path = f'crates/swarm-agents/src/telemetry/{fname}'
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace('use super::*;', 'use super::*;\n    use crate::telemetry::*;')
    with open(path, 'w') as f:
        f.write(content)
