import re

with open('crates/swarm-agents/src/telemetry.rs.bak', 'r') as f:
    lines = f.readlines()

ib_start = 321
ib_end = 340
ib_lines = "".join(lines[ib_start:ib_end+1])

# Write IterationBuilder into types.rs
with open('crates/swarm-agents/src/telemetry/types.rs', 'a') as f:
    f.write("\n" + ib_lines)
