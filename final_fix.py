import re

with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

text = text.replace('pub struct EvaluatorSnapshot', '#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct EvaluatorSnapshot')
text = text.replace('pub struct HarnessComponentTrace', '#[derive(Default)]\npub struct HarnessComponentTrace')

# ExecutionArtifact initialization missing fields!
# In our earlier script we pulled ExecutionArtifact methods out maybe incorrectly or the fields were removed?
# Let's see where ExecutionArtifact is defined in types.rs.
