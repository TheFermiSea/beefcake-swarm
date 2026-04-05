import os

with open('/tmp/telemetry.rs', 'r') as f:
    lines = f.readlines()

def get_lines(ranges):
    out = []
    for s, e in ranges:
        start = s
        while start > 0 and (lines[start-1].strip() == '' or lines[start-1].strip().startswith('//') or lines[start-1].strip().startswith('#[')):
            start -= 1
        out.append("".join(lines[start:e+1]))
    return "\n".join(out)

# We use the previous line ranges to extract blocks.

types_blocks = [
    (12, 25), (26, 38), (44, 46), (47, 60), (61, 73), (74, 84), (85, 103),
    (104, 117), (118, 134), (135, 155), (156, 168), (169, 174), (175, 205),
    (206, 256), (257, 290), (291, 297), (783, 797), (1054, 1065), (1066, 1080),
    (1081, 1093), (1094, 1104), (1105, 1115), (1116, 1137), (1273, 1286)
]

aggregation_blocks = [
    (298, 318), (342, 703), (903, 920), (921, 932), (933, 947), (948, 965),
    (966, 970), (971, 1049), (1138, 1231), (1232, 1268), (1287, 1299),
    (1300, 1365), (1370, 1402)
]

export_blocks = [
    (704, 715), (716, 742), (743, 782), (798, 820), (821, 868), (869, 902)
]

events_blocks = [
    (1407, 1424), (1425, 1457), (1458, 1467), (1468, 1645)
]

# Write to disk
os.makedirs('crates/swarm-agents/src/telemetry', exist_ok=True)

header = """use super::*;
use std::path::Path;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

"""

with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write(header + get_lines(types_blocks))

with open('crates/swarm-agents/src/telemetry/aggregation.rs', 'w') as f:
    f.write(header + get_lines(aggregation_blocks))

with open('crates/swarm-agents/src/telemetry/export.rs', 'w') as f:
    f.write(header + get_lines(export_blocks))

with open('crates/swarm-agents/src/telemetry/events.rs', 'w') as f:
    f.write(header + get_lines(events_blocks))

with open('crates/swarm-agents/src/telemetry/init.rs', 'w') as f:
    f.write(header + "// Initialization logic will go here if needed.\n")
