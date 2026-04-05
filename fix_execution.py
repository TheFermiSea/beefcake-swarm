with open('crates/swarm-agents/src/telemetry/types.rs', 'r') as f:
    text = f.read()

# E0063: missing fields evaluator_snapshot, retry_rationale, route_decision, verifier_snapshot
# in ExecutionArtifact::new
text = text.replace(
"""        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
        }""",
"""        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            route_decision: None,
            verifier_snapshot: None,
            evaluator_snapshot: None,
            retry_rationale: None,
        }"""
)

with open('crates/swarm-agents/src/telemetry/types.rs', 'w') as f:
    f.write(text)
