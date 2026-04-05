import re

with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

# the regex didn't match the duplicates correctly. I will use a simple string split to remove the second block
# Look at lines 42-65 where RouteDecision is defined twice.
lines = text.split('\n')
new_lines = []
in_second_route_decision = False
seen_first = False

for line in lines:
    if line.startswith('pub struct RouteDecision {'):
        if not seen_first:
            seen_first = True
            new_lines.append(line)
        else:
            in_second_route_decision = True
            # Also need to drop the derive above it
            if new_lines[-1] == '#[derive(Debug, Clone, Serialize, Deserialize)]':
                new_lines.pop()
    elif in_second_route_decision:
        if line == '}':
            in_second_route_decision = False
    else:
        new_lines.append(line)

with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write('\n'.join(new_lines))
