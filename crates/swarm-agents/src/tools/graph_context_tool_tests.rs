use super::*;
use coordination::context_packer::SemanticCodeGraph;
use rig::tool::Tool;
use std::path::Path;
use std::sync::Arc;

fn build_test_graph() -> Arc<SemanticCodeGraph> {
    let source = r#"
    fn callee() -> u32 { 42 }
    fn caller() -> u32 { callee() }
    fn a() -> u32 { b() }
    fn b() -> u32 { c() }
    fn c() -> u32 { 42 }
    "#;
    Arc::new(SemanticCodeGraph::from_sources(&[("test.rs", source)]))
}

async fn run_tool_test(
    target: &str,
    hops: Option<u32>,
    kind: Option<&str>,
    expected_contents: &[&str],
) {
    let graph = build_test_graph();
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);
    let input = GraphContextInput {
        target: target.to_string(),
        hops,
        kind: kind.map(|s| s.to_string()),
    };
    let result = tool.call(input).await.unwrap();
    for expected in expected_contents {
        assert!(
            result.contains(expected),
            "Result did not contain expected string '{}'\nResult:\n{}",
            expected,
            result
        );
    }
}

#[tokio::test]
async fn test_graph_context_queries() {
    // success
    run_tool_test("caller", Some(2), Some("callees"), &["caller", "callee"]).await;
    // no_hops_default
    run_tool_test("caller", None, None, &["caller", "callee"]).await;
    // not_found
    run_tool_test(
        "nonexistent",
        Some(1),
        Some("callers"),
        &["Graph query error", "nonexistent"],
    )
    .await;
    // truncation
    run_tool_test(
        "a",
        Some(1),
        Some("callees"),
        &["Nodes:", "Edges:", "a", "b"],
    )
    .await;
}

#[tokio::test]
async fn test_graph_context_definition() {
    let graph = build_test_graph();
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);
    let def = tool.definition("".to_string()).await;
    assert_eq!(def.name, "graph_context");
    assert!(def.description.contains("semantic code dependency graph"));
}

#[test]
fn test_parse_query_kind() {
    use coordination::reviewer_tools::graph_rag::QueryKind;
    assert_eq!(parse_query_kind("callers"), QueryKind::Callers);
    assert_eq!(parse_query_kind("callees"), QueryKind::Callees);
    assert_eq!(parse_query_kind("implementors"), QueryKind::Implementors);
    assert_eq!(parse_query_kind("dependencies"), QueryKind::Dependencies);
    assert_eq!(parse_query_kind("dependents"), QueryKind::Dependents);
    assert_eq!(parse_query_kind("unknown"), QueryKind::Callees); // Default
    assert_eq!(parse_query_kind("CALLERS"), QueryKind::Callers); // Case insensitive
}

#[tokio::test]
async fn test_graph_context_error_handling() {
    let graph = Arc::new(SemanticCodeGraph::from_sources(&[]));
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);

    let input = GraphContextInput {
        target: "anything".to_string(),
        hops: Some(1),
        kind: Some("callees".to_string()),
    };

    let result = tool.call(input).await.unwrap();
    assert!(result.contains("Graph query error"));
}
