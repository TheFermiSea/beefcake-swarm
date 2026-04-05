import re

with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

# Add #[derive(Debug, Clone, Serialize, Deserialize)] to SessionMetrics, FailureLedgerEntry, HarnessComponentTrace, ValidationMetric

to_derive = [
    'pub struct SessionMetrics',
    'pub struct FailureLedgerEntry',
    'pub struct HarnessComponentTrace',
    'pub struct ValidationMetric',
    'pub struct IterationBuilder'
]

for s in to_derive:
    text = text.replace(s, '#[derive(Debug, Clone, Serialize, Deserialize)]\n' + s)

# IterationBuilder had pub fields issues when we did regex replace. Let's fix pub.
# the script `sed -i 's/    \(.*[a-z_0-9]\+:[ \t]*[A-Za-z_0-9<>, ]\+\),/    pub \1,/g' crates/swarm-agents/src/telemetry/types.rs`
# messed up `impl SessionMetrics` somehow or others.

with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write(text)
