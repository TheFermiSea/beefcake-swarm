import re
import json

def get_ranges():
    with open('/tmp/telemetry.rs', 'r') as f:
        lines = f.readlines()

    blocks = []
    current_block = None
    nesting = 0

    for i, line in enumerate(lines):
        line = line.strip()
        # Find start of pub item or impl
        if (line.startswith('pub struct') or line.startswith('pub enum') or
            line.startswith('pub fn') or line.startswith('pub const') or
            line.startswith('pub mod') or line.startswith('impl ') or
            line.startswith('fn ') or line.startswith('#[cfg(test)]') or line.startswith('pub(crate) fn')):

            # Start tracking a block
            if current_block is None and not line.endswith(';'):
                start = i
                while start > 0 and (lines[start-1].strip().startswith('#[') or lines[start-1].strip().startswith('///') or lines[start-1].strip() == ''):
                    start -= 1
                current_block = {'start': start, 'type': line.split('{')[0].strip()}

        # Handle nesting
        nesting += line.count('{')
        nesting -= line.count('}')

        if current_block is not None and nesting == 0 and line.count('}') > 0:
            current_block['end'] = i
            blocks.append(current_block)
            current_block = None

        # special case for consts ending with ;
        if current_block is None and line.startswith('pub const') and line.endswith(';'):
            start = i
            while start > 0 and (lines[start-1].strip().startswith('#[') or lines[start-1].strip().startswith('///')):
                start -= 1
            blocks.append({'start': start, 'end': i, 'type': line})

    # For testing modules we need special care, let's just group everything from line 1490 onwards into tests if we want to
    # Actually the instruction says:
    # crates/swarm-agents/src/telemetry/types.rs — all structs, enums, and type aliases
    # crates/swarm-agents/src/telemetry/events.rs — event emission functions and span helpers
    # crates/swarm-agents/src/telemetry/export.rs — exporter configuration and sink setup
    # crates/swarm-agents/src/telemetry/aggregation.rs — buffering and aggregation logic
    # crates/swarm-agents/src/telemetry/init.rs — initialization, shutdown, and global state
    pass
