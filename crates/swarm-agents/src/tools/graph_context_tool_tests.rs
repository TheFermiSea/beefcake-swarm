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

#[tokio::test]
async fn test_graph_context_success() {
    let graph = build_test_graph();
    let working_dir = Path::new("/tmp");
    let tool = GraphContextTool::new(working_dir, graph);

    let input = GraphContextInput {
        target: "caller".to_string(),
        hops: Some(2),
        kind: Some("callees".to_string()),
    };

    let result = tool.call(input).await.unwrap();
    assert!(result.contains("caller"));
    assert!(result.contains("callee"));
}

#[tokio::test]
async fn test_graph_context_no_hops_default() {
    let graph = build_test_graph();
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);

    let input = GraphContextInput {
        target: "caller".to_string(),
        hops: None,
        kind: None, // should default to "callees"
    };

    let result = tool.call(input).await.unwrap();
    assert!(result.contains("caller"));
    assert!(result.contains("callee"));
}

#[tokio::test]
async fn test_graph_context_not_found() {
    let graph = build_test_graph();
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);

    let input = GraphContextInput {
        target: "nonexistent".to_string(),
        hops: Some(1),
        kind: Some("callers".to_string()),
    };

    let result = tool.call(input).await.unwrap();
    assert!(result.contains("Graph query error"));
    assert!(result.contains("nonexistent"));
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
async fn test_graph_context_truncation() {
    let graph = build_test_graph();
    let tool = GraphContextTool::new(Path::new("/tmp"), graph);

    let input = GraphContextInput {
        target: "a".to_string(),
        hops: Some(1), // limit depth to test edge counts
        kind: Some("callees".to_string()),
    };

    let result = tool.call(input).await.unwrap();
    assert!(result.contains("Nodes:"));
    assert!(result.contains("Edges:"));
    assert!(result.contains("a"));
    assert!(result.contains("b"));
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
