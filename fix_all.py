import re

with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

# remove conflicting derives
text = re.sub(r'#\[derive\(.*?\)\]\n#\[derive\(.*?\)\]', '#[derive(Debug, Clone, Serialize, Deserialize)]', text)
text = text.replace('#[derive(Debug, Clone, Default, Serialize, Deserialize)]\n#[derive(Debug, Clone, Serialize, Deserialize)]', '#[derive(Debug, Clone, Default, Serialize, Deserialize)]')

# The error also said EvaluatorSnapshot doesn't have derive. Wait, EvaluatorSnapshot was already in the original file!
# Let me look at EvaluatorSnapshot in types.rs.
with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write(text)
