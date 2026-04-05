with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

import re

# Remove duplicate RouteDecision definition
text = re.sub(r'#\[derive\(Debug, Clone, Serialize, Deserialize\)\]\npub struct RouteDecision \{.*?\n\}\n\n#\[derive\(Debug, Clone, Serialize, Deserialize\)\]\npub struct RouteDecision \{.*?\n\}\n',
              r'#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct RouteDecision {\n    pub coder: String,\n    pub input_error_categories: Vec<String>,\n    pub tier: String,\n    #[serde(default, skip_serializing_if = "Option::is_none")]\n    pub rationale: Option<String>,\n}\n', text, flags=re.DOTALL)

# Make IterationBuilder public
text = text.replace('struct IterationBuilder {', 'pub struct IterationBuilder {')
# Make fields public
fields = ['iteration: u32', 'tier: String', 'agent_model: String', 'agent_prompt_tokens: u32', 'agent_completion_tokens: u32', 'agent_response_ms: u64', 'verifier_ms: u64', 'error_count: usize', 'error_categories: Vec<String>', 'no_change: bool', 'auto_fix_applied: bool', 'regression_detected: bool', 'rollback_performed: bool', 'escalated: bool', 'coder_route: Option<String>', 'artifacts: Vec<ArtifactRecord>', 'execution_artifact: ExecutionArtifact', 'progress_score: Option<f64>', 'best_error_count: Option<usize>']
for field in fields:
    text = text.replace(f'    {field},', f'    pub {field},')

with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write(text)
