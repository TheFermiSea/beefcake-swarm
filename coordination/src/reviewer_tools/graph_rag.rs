//! GraphRAG / CocoIndex tool wrapper — bounded dependency and impact queries.
//!
//! Wraps a graph-based code index (e.g., CocoIndex) with structured input/output,
//! timeout enforcement, and result truncation for safe use by the reviewer agent.

use serde::{Deserialize, Serialize};

/// Configuration for the GraphRAG runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagConfig {
    /// Base URL of the CocoIndex / graph query endpoint.
    pub endpoint_url: String,
    /// Maximum execution time in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of results to return per query.
    pub max_results: usize,
    /// Maximum depth for transitive dependency traversal.
    pub max_depth: u32,
    /// Whether to include test files in results.
    pub include_tests: bool,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            endpoint_url: "http://localhost:8300".to_string(),
            timeout_ms: 15_000,
            max_results: 50,
            max_depth: 5,
            include_tests: false,
        }
    }
}

/// The kind of graph query to execute.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueryKind {
    /// Find all callers of a function/method.
    Callers,
    /// Find all callees of a function/method.
    Callees,
    /// Find all implementors of a trait.
    Implementors,
    /// Find all usages of a type/struct.
    TypeUsages,
    /// Find transitive dependencies of a module/file.
    Dependencies,
    /// Find modules/files that depend on this one (reverse deps).
    Dependents,
    /// Impact analysis: what could break if this symbol changes.
    ImpactAnalysis,
}

impl std::fmt::Display for QueryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryKind::Callers => write!(f, "callers"),
            QueryKind::Callees => write!(f, "callees"),
            QueryKind::Implementors => write!(f, "implementors"),
            QueryKind::TypeUsages => write!(f, "type_usages"),
            QueryKind::Dependencies => write!(f, "dependencies"),
            QueryKind::Dependents => write!(f, "dependents"),
            QueryKind::ImpactAnalysis => write!(f, "impact_analysis"),
        }
    }
}

/// A query for the graph-based code index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagQuery {
    /// The symbol or path to query about.
    pub target: String,
    /// Kind of query to execute.
    pub kind: QueryKind,
    /// Optional scope restriction (e.g., crate name or path prefix).
    pub scope: Option<String>,
    /// Maximum traversal depth (overrides config if set).
    pub max_depth: Option<u32>,
    /// Language filter (rust, typescript, python, etc.).
    pub language: String,
}

impl GraphRagQuery {
    /// Create a callers query.
    pub fn callers(symbol: &str, language: &str) -> Self {
        Self {
            target: symbol.to_string(),
            kind: QueryKind::Callers,
            scope: None,
            max_depth: None,
            language: language.to_string(),
        }
    }

    /// Create a callees query.
    pub fn callees(symbol: &str, language: &str) -> Self {
        Self {
            target: symbol.to_string(),
            kind: QueryKind::Callees,
            scope: None,
            max_depth: None,
            language: language.to_string(),
        }
    }

    /// Create an implementors query (for traits/interfaces).
    pub fn implementors(trait_name: &str, language: &str) -> Self {
        Self {
            target: trait_name.to_string(),
            kind: QueryKind::Implementors,
            scope: None,
            max_depth: None,
            language: language.to_string(),
        }
    }

    /// Create an impact analysis query.
    pub fn impact(symbol: &str, language: &str) -> Self {
        Self {
            target: symbol.to_string(),
            kind: QueryKind::ImpactAnalysis,
            scope: None,
            max_depth: None,
            language: language.to_string(),
        }
    }

    /// Restrict query to a scope.
    pub fn in_scope(mut self, scope: &str) -> Self {
        self.scope = Some(scope.to_string());
        self
    }

    /// Set maximum traversal depth.
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Effective depth considering query override and config default.
    pub fn effective_depth(&self, config_default: u32) -> u32 {
        self.max_depth.unwrap_or(config_default)
    }
}

/// A single node in the graph result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Fully qualified symbol name.
    pub symbol: String,
    /// File path where the symbol is defined.
    pub file: String,
    /// Line number (1-indexed).
    pub line: u32,
    /// The kind of symbol (function, struct, trait, module, etc.).
    pub symbol_kind: String,
    /// Depth from the query target (0 = direct).
    pub depth: u32,
}

impl std::fmt::Display for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{} {} [{}] (depth={})",
            self.file, self.line, self.symbol, self.symbol_kind, self.depth
        )
    }
}

/// An edge in the graph result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source symbol.
    pub from: String,
    /// Target symbol.
    pub to: String,
    /// Relationship type (calls, implements, uses, depends_on).
    pub relation: String,
}

/// Result from a GraphRAG query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagResult {
    /// Nodes matching the query.
    pub nodes: Vec<GraphNode>,
    /// Edges between nodes (if requested).
    pub edges: Vec<GraphEdge>,
    /// Total results before truncation.
    pub total_results: usize,
    /// Whether results were truncated.
    pub truncated: bool,
    /// Maximum depth actually traversed.
    pub depth_reached: u32,
    /// Execution time in milliseconds.
    pub execution_ms: u64,
    /// Whether the query timed out.
    pub timed_out: bool,
    /// Error message (if any).
    pub error: Option<String>,
}

impl GraphRagResult {
    /// Create a successful result.
    pub fn ok(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>, execution_ms: u64) -> Self {
        let total = nodes.len();
        let depth_reached = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        Self {
            nodes,
            edges,
            total_results: total,
            truncated: false,
            depth_reached,
            execution_ms,
            timed_out: false,
            error: None,
        }
    }

    /// Create a timeout result.
    pub fn timeout(timeout_ms: u64) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            total_results: 0,
            truncated: false,
            depth_reached: 0,
            execution_ms: timeout_ms,
            timed_out: true,
            error: Some(format!("timed out after {}ms", timeout_ms)),
        }
    }

    /// Create an error result.
    pub fn err(error: &str, execution_ms: u64) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            total_results: 0,
            truncated: false,
            depth_reached: 0,
            execution_ms,
            timed_out: false,
            error: Some(error.to_string()),
        }
    }

    /// Whether the query was successful.
    pub fn is_success(&self) -> bool {
        self.error.is_none() && !self.timed_out
    }

    /// Truncate to max results.
    pub fn truncate_to(&mut self, max: usize) {
        if self.nodes.len() > max {
            self.total_results = self.nodes.len();
            self.nodes.truncate(max);
            self.truncated = true;
        }
    }

    /// Get nodes at a specific depth.
    pub fn at_depth(&self, depth: u32) -> Vec<&GraphNode> {
        self.nodes.iter().filter(|n| n.depth == depth).collect()
    }

    /// Get unique files referenced in results.
    pub fn affected_files(&self) -> Vec<&str> {
        let mut files: Vec<&str> = self.nodes.iter().map(|n| n.file.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        files
    }

    /// Compact summary line.
    pub fn summary_line(&self) -> String {
        if let Some(ref err) = self.error {
            format!("[ERROR] {} ({}ms)", err, self.execution_ms)
        } else if self.truncated {
            format!(
                "[OK] {} nodes (truncated from {}, depth={}, {}ms)",
                self.nodes.len(),
                self.total_results,
                self.depth_reached,
                self.execution_ms
            )
        } else {
            format!(
                "[OK] {} nodes (depth={}, {}ms)",
                self.nodes.len(),
                self.depth_reached,
                self.execution_ms
            )
        }
    }
}

/// Runner that manages GraphRAG query execution.
pub struct GraphRagRunner {
    config: GraphRagConfig,
}

impl GraphRagRunner {
    /// Create a new runner with default config.
    pub fn new() -> Self {
        Self {
            config: GraphRagConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: GraphRagConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &GraphRagConfig {
        &self.config
    }

    /// Validate a query before execution.
    pub fn validate_query(&self, query: &GraphRagQuery) -> Result<(), String> {
        if query.target.is_empty() {
            return Err("target symbol is required".to_string());
        }
        if query.language.is_empty() {
            return Err("language is required".to_string());
        }
        if let Some(depth) = query.max_depth {
            if depth > 20 {
                return Err(format!("max_depth {} exceeds limit of 20", depth));
            }
        }
        Ok(())
    }

    /// Apply config bounds to a result.
    pub fn apply_bounds(&self, mut result: GraphRagResult) -> GraphRagResult {
        result.truncate_to(self.config.max_results);
        result
    }

    /// Filter out test files from results (if config says to exclude them).
    pub fn filter_tests(&self, mut result: GraphRagResult) -> GraphRagResult {
        if !self.config.include_tests {
            let original_len = result.nodes.len();
            result
                .nodes
                .retain(|n| !n.file.contains("/tests/") && !n.file.starts_with("tests/"));
            if result.nodes.len() < original_len {
                result.total_results = original_len;
                result.truncated = true;
            }
        }
        result
    }
}

impl Default for GraphRagRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callers_query() {
        let query = GraphRagQuery::callers("MyStruct::process", "rust");
        assert_eq!(query.target, "MyStruct::process");
        assert_eq!(query.kind, QueryKind::Callers);
        assert_eq!(query.language, "rust");
        assert!(query.scope.is_none());
    }

    #[test]
    fn test_impact_query_with_scope() {
        let query = GraphRagQuery::impact("SwarmMemory::compact", "rust")
            .in_scope("coordination")
            .with_depth(3);
        assert_eq!(query.kind, QueryKind::ImpactAnalysis);
        assert_eq!(query.scope.as_deref(), Some("coordination"));
        assert_eq!(query.max_depth, Some(3));
        assert_eq!(query.effective_depth(5), 3);
    }

    #[test]
    fn test_effective_depth_default() {
        let query = GraphRagQuery::callers("foo", "rust");
        assert_eq!(query.effective_depth(5), 5);
    }

    #[test]
    fn test_result_ok() {
        let nodes = vec![
            GraphNode {
                symbol: "caller_a".to_string(),
                file: "src/a.rs".to_string(),
                line: 10,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
            GraphNode {
                symbol: "caller_b".to_string(),
                file: "src/b.rs".to_string(),
                line: 20,
                symbol_kind: "function".to_string(),
                depth: 2,
            },
        ];
        let edges = vec![GraphEdge {
            from: "caller_a".to_string(),
            to: "target".to_string(),
            relation: "calls".to_string(),
        }];
        let result = GraphRagResult::ok(nodes, edges, 120);
        assert!(result.is_success());
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.depth_reached, 2);
        assert!(!result.truncated);
        assert!(result.summary_line().contains("[OK]"));
        assert!(result.summary_line().contains("2 nodes"));
    }

    #[test]
    fn test_result_truncation() {
        let nodes: Vec<GraphNode> = (0..30)
            .map(|i| GraphNode {
                symbol: format!("sym_{}", i),
                file: format!("src/file{}.rs", i),
                line: i as u32,
                symbol_kind: "function".to_string(),
                depth: (i % 3) as u32,
            })
            .collect();
        let mut result = GraphRagResult::ok(nodes, vec![], 200);
        result.truncate_to(10);
        assert!(result.truncated);
        assert_eq!(result.nodes.len(), 10);
        assert_eq!(result.total_results, 30);
        assert!(result.summary_line().contains("truncated from 30"));
    }

    #[test]
    fn test_result_timeout() {
        let result = GraphRagResult::timeout(15000);
        assert!(!result.is_success());
        assert!(result.timed_out);
        assert!(result.summary_line().contains("ERROR"));
    }

    #[test]
    fn test_result_error() {
        let result = GraphRagResult::err("index not available", 0);
        assert!(!result.is_success());
        assert!(result.summary_line().contains("index not available"));
    }

    #[test]
    fn test_at_depth() {
        let nodes = vec![
            GraphNode {
                symbol: "direct".to_string(),
                file: "a.rs".to_string(),
                line: 1,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
            GraphNode {
                symbol: "transitive".to_string(),
                file: "b.rs".to_string(),
                line: 2,
                symbol_kind: "function".to_string(),
                depth: 2,
            },
            GraphNode {
                symbol: "also_direct".to_string(),
                file: "c.rs".to_string(),
                line: 3,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
        ];
        let result = GraphRagResult::ok(nodes, vec![], 50);
        assert_eq!(result.at_depth(1).len(), 2);
        assert_eq!(result.at_depth(2).len(), 1);
        assert_eq!(result.at_depth(3).len(), 0);
    }

    #[test]
    fn test_affected_files() {
        let nodes = vec![
            GraphNode {
                symbol: "a".to_string(),
                file: "src/lib.rs".to_string(),
                line: 1,
                symbol_kind: "function".to_string(),
                depth: 0,
            },
            GraphNode {
                symbol: "b".to_string(),
                file: "src/main.rs".to_string(),
                line: 2,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
            GraphNode {
                symbol: "c".to_string(),
                file: "src/lib.rs".to_string(),
                line: 10,
                symbol_kind: "struct".to_string(),
                depth: 1,
            },
        ];
        let result = GraphRagResult::ok(nodes, vec![], 30);
        let files = result.affected_files();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"src/lib.rs"));
        assert!(files.contains(&"src/main.rs"));
    }

    #[test]
    fn test_node_display() {
        let node = GraphNode {
            symbol: "foo::bar".to_string(),
            file: "src/foo.rs".to_string(),
            line: 42,
            symbol_kind: "function".to_string(),
            depth: 1,
        };
        assert_eq!(
            node.to_string(),
            "src/foo.rs:42 foo::bar [function] (depth=1)"
        );
    }

    #[test]
    fn test_query_kind_display() {
        assert_eq!(QueryKind::Callers.to_string(), "callers");
        assert_eq!(QueryKind::ImpactAnalysis.to_string(), "impact_analysis");
        assert_eq!(QueryKind::Dependents.to_string(), "dependents");
    }

    #[test]
    fn test_validate_query() {
        let runner = GraphRagRunner::new();

        // Valid query
        let q = GraphRagQuery::callers("foo", "rust");
        assert!(runner.validate_query(&q).is_ok());

        // Empty target
        let q = GraphRagQuery::callers("", "rust");
        assert!(runner.validate_query(&q).is_err());

        // Empty language
        let q = GraphRagQuery {
            target: "foo".to_string(),
            kind: QueryKind::Callers,
            scope: None,
            max_depth: None,
            language: String::new(),
        };
        assert!(runner.validate_query(&q).is_err());

        // Excessive depth
        let q = GraphRagQuery::callers("foo", "rust").with_depth(25);
        assert!(runner.validate_query(&q).is_err());
    }

    #[test]
    fn test_apply_bounds() {
        let runner = GraphRagRunner::with_config(GraphRagConfig {
            max_results: 5,
            ..Default::default()
        });
        let nodes: Vec<GraphNode> = (0..20)
            .map(|i| GraphNode {
                symbol: format!("sym_{}", i),
                file: format!("file{}.rs", i),
                line: i,
                symbol_kind: "function".to_string(),
                depth: 0,
            })
            .collect();
        let result = GraphRagResult::ok(nodes, vec![], 100);
        let bounded = runner.apply_bounds(result);
        assert_eq!(bounded.nodes.len(), 5);
        assert!(bounded.truncated);
    }

    #[test]
    fn test_filter_tests() {
        let runner = GraphRagRunner::new(); // include_tests = false by default
        let nodes = vec![
            GraphNode {
                symbol: "production".to_string(),
                file: "src/lib.rs".to_string(),
                line: 1,
                symbol_kind: "function".to_string(),
                depth: 0,
            },
            GraphNode {
                symbol: "test_helper".to_string(),
                file: "tests/integration.rs".to_string(),
                line: 5,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
            GraphNode {
                symbol: "unit_test".to_string(),
                file: "src/foo/tests/bar.rs".to_string(),
                line: 10,
                symbol_kind: "function".to_string(),
                depth: 1,
            },
        ];
        let result = GraphRagResult::ok(nodes, vec![], 50);
        let filtered = runner.filter_tests(result);
        assert_eq!(filtered.nodes.len(), 1);
        assert_eq!(filtered.nodes[0].symbol, "production");
        assert!(filtered.truncated);
    }

    #[test]
    fn test_config_serde() {
        let config = GraphRagConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: GraphRagConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.endpoint_url, "http://localhost:8300");
        assert_eq!(parsed.timeout_ms, 15_000);
        assert_eq!(parsed.max_results, 50);
        assert_eq!(parsed.max_depth, 5);
    }

    #[test]
    fn test_query_serde() {
        let query = GraphRagQuery::impact("MyTrait", "rust").in_scope("crate_a");
        let json = serde_json::to_string(&query).unwrap();
        let parsed: GraphRagQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.target, "MyTrait");
        assert_eq!(parsed.kind, QueryKind::ImpactAnalysis);
        assert_eq!(parsed.scope.as_deref(), Some("crate_a"));
    }
}
