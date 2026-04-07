//! Semantic Code Graph — petgraph-backed dependency graph from tree-sitter AST parsing.
//!
//! Builds a directed graph of code symbols (functions, structs, traits, impls) and
//! their relationships (calls, implements, instantiates, module containment).
//! Supports BFS extraction for Graph RAG context injection.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use tracing::warn;
use tree_sitter::{Node, Parser};

use super::ast_index::SymbolKind;
use super::file_walker::FileWalker;

/// Data attached to each node in the semantic graph.
#[derive(Debug, Clone)]
pub struct NodeData {
    /// Symbol name, e.g. "MyStruct::method" or "free_function"
    pub symbol: String,
    /// Relative file path
    pub file: String,
    /// Line number (0-indexed)
    pub line: u32,
    /// Symbol kind (reused from ast_index)
    pub kind: SymbolKind,
}

/// Kinds of edges in the semantic graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    /// Function/method call relationship
    Calls,
    /// Trait implementation (impl Trait for Type)
    Implements,
    /// Struct/enum instantiation (e.g. Foo { ... } or Foo::new())
    Instantiates,
    /// Module containment (mod contains symbol)
    ModuleContains,
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeKind::Calls => write!(f, "calls"),
            EdgeKind::Implements => write!(f, "implements"),
            EdgeKind::Instantiates => write!(f, "instantiates"),
            EdgeKind::ModuleContains => write!(f, "contains"),
        }
    }
}

/// A petgraph-backed semantic code dependency graph.
pub struct SemanticCodeGraph {
    graph: DiGraph<NodeData, EdgeKind>,
    index: HashMap<String, NodeIndex>,
}

impl SemanticCodeGraph {
    /// Build a semantic graph by walking `.rs` files under `worktree`.
    pub fn build(worktree: &Path) -> Self {
        let walker = FileWalker::new(worktree);
        let rs_files = walker.rust_files();

        let mut scg = Self {
            graph: DiGraph::new(),
            index: HashMap::new(),
        };

        // Collect (relative_path, source) pairs
        let mut sources: Vec<(String, String)> = Vec::new();
        for path in &rs_files {
            let rel = path
                .strip_prefix(worktree)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            match std::fs::read_to_string(path) {
                Ok(src) => sources.push((rel, src)),
                Err(e) => {
                    warn!("Failed to read {}: {}", path.display(), e);
                }
            }
        }

        let refs: Vec<(&str, &str)> = sources
            .iter()
            .map(|(f, s)| (f.as_str(), s.as_str()))
            .collect();
        scg.populate_from_sources(&refs);
        scg
    }

    /// Build a semantic graph from in-memory source pairs (for testing).
    pub fn from_sources(files: &[(&str, &str)]) -> Self {
        let mut scg = Self {
            graph: DiGraph::new(),
            index: HashMap::new(),
        };
        scg.populate_from_sources(files);
        scg
    }

    /// Internal: populate graph from source file pairs.
    fn populate_from_sources(&mut self, files: &[(&str, &str)]) {
        let mut parser = Parser::new();
        if parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .is_err()
        {
            warn!("Failed to set tree-sitter-rust language");
            return;
        }

        // First pass: extract declared symbols and insert as nodes.
        // Also collect per-file parse trees for the second pass.
        let mut trees: Vec<(&str, &str, tree_sitter::Tree)> = Vec::new();

        for &(file, source) in files {
            match parser.parse(source, None) {
                Some(tree) => {
                    self.extract_declarations(file, source, tree.root_node());
                    trees.push((file, source, tree));
                }
                None => {
                    warn!("tree-sitter failed to parse {}", file);
                }
            }
        }

        // Second pass: extract edges (calls, implements, instantiates).
        for (file, source, tree) in &trees {
            self.extract_edges(file, source, tree.root_node());
        }
    }

    /// First pass: walk the AST to find declared symbols and insert graph nodes.
    fn extract_declarations(&mut self, file: &str, source: &str, root: Node) {
        let source_bytes = source.as_bytes();
        self.walk_for_declarations(file, source_bytes, root, None);
    }

    fn walk_for_declarations(
        &mut self,
        file: &str,
        source: &[u8],
        node: Node,
        parent_name: Option<&str>,
    ) {
        match node.kind() {
            "function_item" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                {
                    let symbol = match parent_name {
                        Some(p) => format!("{}::{}", p, name),
                        None => name.to_string(),
                    };
                    self.insert_node(
                        file,
                        &symbol,
                        node.start_position().row as u32,
                        SymbolKind::Function,
                    );
                }
            }
            "struct_item" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                {
                    self.insert_node(
                        file,
                        name,
                        node.start_position().row as u32,
                        SymbolKind::Struct,
                    );
                }
            }
            "enum_item" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                {
                    self.insert_node(
                        file,
                        name,
                        node.start_position().row as u32,
                        SymbolKind::Enum,
                    );
                }
            }
            "trait_item" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                {
                    self.insert_node(
                        file,
                        name,
                        node.start_position().row as u32,
                        SymbolKind::Trait,
                    );
                }
            }
            "impl_item" => {
                // Extract the impl target type name for use as parent context.
                let impl_name = self.extract_impl_type_name(node, source);
                // Walk children with impl type as parent.
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk_for_declarations(file, source, child, impl_name.as_deref());
                }
                return; // Don't walk children again below.
            }
            "mod_item" => {
                if let Some(name) = node
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(source).ok())
                {
                    self.insert_node(
                        file,
                        name,
                        node.start_position().row as u32,
                        SymbolKind::Mod,
                    );
                }
            }
            _ => {}
        }

        // Walk children (except for impl_item which is handled above).
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk_for_declarations(file, source, child, parent_name);
        }
    }

    /// Second pass: walk the AST to find edges (calls, implements, instantiates).
    fn extract_edges(&mut self, file: &str, source: &str, root: Node) {
        let source_bytes = source.as_bytes();
        self.walk_for_edges(file, source_bytes, root);
    }

    fn walk_for_edges(&mut self, file: &str, source: &[u8], node: Node) {
        match node.kind() {
            // Free function call: foo()
            "call_expression" => {
                if let Some(func_node) = node.child_by_field_name("function") {
                    let caller = self.find_enclosing_function(file, node);
                    match func_node.kind() {
                        // Direct call: my_func(...)
                        "identifier" => {
                            if let Ok(callee_name) = func_node.utf8_text(source) {
                                if let Some(caller_key) = caller {
                                    self.add_call_edge(&caller_key, callee_name);
                                }
                            }
                        }
                        // Scoped call: Foo::bar(...)
                        "scoped_identifier" | "field_expression" => {
                            if let Ok(callee_text) = func_node.utf8_text(source) {
                                if let Some(caller_key) = caller {
                                    // For scoped identifiers like Foo::bar, try to resolve
                                    self.add_call_edge(&caller_key, callee_text);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            // impl Trait for Type — Implements edge
            "impl_item" => {
                self.extract_impl_edges(file, source, node);
            }
            // Struct expression: Foo { field: value } — Instantiates edge
            "struct_expression" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(struct_name) = name_node.utf8_text(source) {
                        let caller = self.find_enclosing_function(file, node);
                        if let Some(caller_key) = caller {
                            self.add_edge_by_name(&caller_key, struct_name, EdgeKind::Instantiates);
                        }
                    }
                }
            }
            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk_for_edges(file, source, child);
        }
    }

    /// Extract the type name from an impl item (the type being implemented).
    fn extract_impl_type_name(&self, node: Node, source: &[u8]) -> Option<String> {
        // impl_item has a "type" field for the implementing type.
        node.child_by_field_name("type")
            .and_then(|n| n.utf8_text(source).ok())
            .map(|s| s.to_string())
    }

    /// Extract Implements edges from an impl_item node.
    fn extract_impl_edges(&mut self, file: &str, source: &[u8], node: Node) {
        // Check if this is `impl Trait for Type` by looking for the "trait" field.
        let trait_name = node
            .child_by_field_name("trait")
            .and_then(|n| n.utf8_text(source).ok())
            .map(|s| s.to_string());

        let type_name = self.extract_impl_type_name(node, source);

        if let (Some(trait_name), Some(type_name)) = (trait_name, type_name) {
            // Type implements Trait
            self.add_edge_by_name_in_file(file, &type_name, &trait_name, EdgeKind::Implements);
        }
    }

    /// Find the enclosing function for a given AST node, returning its graph key.
    fn find_enclosing_function(&self, file: &str, node: Node) -> Option<String> {
        let mut current = node.parent();
        while let Some(parent) = current {
            if parent.kind() == "function_item" {
                if let Some(name_node) = parent.child_by_field_name("name") {
                    if let Ok(fn_name) = name_node.utf8_text(&[]) {
                        // Try to find this function in our index, trying various key forms
                        let key = format!("{}::{}", file, fn_name);
                        if self.index.contains_key(&key) {
                            return Some(key);
                        }
                    }
                }
                // If we can't get the name from empty source, search by line number
                let line = parent.start_position().row as u32;
                // Find the node at this file/line
                for (key, &idx) in &self.index {
                    if key.starts_with(file) {
                        let data = &self.graph[idx];
                        if data.line == line && data.kind == SymbolKind::Function {
                            return Some(key.clone());
                        }
                    }
                }
            }
            current = parent.parent();
        }
        None
    }

    /// Insert a node into the graph.
    fn insert_node(&mut self, file: &str, symbol: &str, line: u32, kind: SymbolKind) {
        let key = format!("{}::{}", file, symbol);
        if self.index.contains_key(&key) {
            return; // Avoid duplicates
        }
        let idx = self.graph.add_node(NodeData {
            symbol: symbol.to_string(),
            file: file.to_string(),
            line,
            kind,
        });
        self.index.insert(key, idx);
    }

    /// Add a Calls edge by resolving callee name across all files.
    fn add_call_edge(&mut self, caller_key: &str, callee_name: &str) {
        let caller_idx = match self.index.get(caller_key) {
            Some(&idx) => idx,
            None => return,
        };

        // Try exact match first (file::callee)
        // Then try any file containing this callee
        let callee_idx = self.resolve_symbol(callee_name);

        if let Some(callee_idx) = callee_idx {
            if caller_idx != callee_idx {
                self.graph.add_edge(caller_idx, callee_idx, EdgeKind::Calls);
            }
        }
    }

    /// Add an edge between two symbols by name, searching across all files.
    fn add_edge_by_name(&mut self, from_key: &str, to_name: &str, kind: EdgeKind) {
        let from_idx = match self.index.get(from_key) {
            Some(&idx) => idx,
            None => return,
        };
        if let Some(to_idx) = self.resolve_symbol(to_name) {
            if from_idx != to_idx {
                self.graph.add_edge(from_idx, to_idx, kind);
            }
        }
    }

    /// Add an edge preferring resolution within the same file.
    fn add_edge_by_name_in_file(
        &mut self,
        file: &str,
        from_name: &str,
        to_name: &str,
        kind: EdgeKind,
    ) {
        // Try file-local resolution first
        let from_key = format!("{}::{}", file, from_name);
        let from_idx = self.index.get(&from_key).or_else(|| {
            self.resolve_symbol(from_name).and_then(|idx| {
                // Verify it exists in our index values
                self.index.values().find(|&&v| v == idx)
            })
        });
        let to_idx = {
            let local_key = format!("{}::{}", file, to_name);
            self.index
                .get(&local_key)
                .copied()
                .or_else(|| self.resolve_symbol(to_name))
        };

        if let (Some(&from_idx), Some(to_idx)) = (from_idx, to_idx) {
            if from_idx != to_idx {
                self.graph.add_edge(from_idx, to_idx, kind);
            }
        }
    }

    /// Resolve a symbol name to a NodeIndex. Handles:
    /// - Exact key match: "file.rs::symbol"
    /// - Suffix match: any key ending in "::symbol"
    /// - Scoped identifiers: "Foo::bar" -> look for key "file.rs::Foo::bar"
    fn resolve_symbol(&self, name: &str) -> Option<NodeIndex> {
        // 1. Exact key match
        if let Some(&idx) = self.index.get(name) {
            return Some(idx);
        }

        // 2. Suffix match: look for keys ending in "::<name>"
        let suffix = format!("::{}", name);
        for (key, &idx) in &self.index {
            if key.ends_with(&suffix) {
                return Some(idx);
            }
        }

        // 3. For scoped names like "Foo::bar", also try substring match
        if name.contains("::") {
            for (key, &idx) in &self.index {
                if key.contains(name) {
                    return Some(idx);
                }
            }
        }

        None
    }

    /// BFS extraction: get a subgraph around a target symbol.
    ///
    /// Uses `QueryKind` from `reviewer_tools::graph_rag` to determine direction:
    /// - `Callers` -> traverse incoming edges (who calls target?)
    /// - `Callees` / others -> traverse outgoing edges (what does target call?)
    pub fn get_subgraph_context(
        &self,
        target: &str,
        hops: u32,
        kind: crate::reviewer_tools::graph_rag::QueryKind,
    ) -> crate::reviewer_tools::graph_rag::GraphRagResult {
        use crate::reviewer_tools::graph_rag::{GraphEdge, GraphNode, GraphRagResult, QueryKind};

        let start = std::time::Instant::now();

        // Find seed nodes matching target (substring match)
        let seeds: Vec<NodeIndex> = self
            .index
            .iter()
            .filter(|(key, _)| key.contains(target))
            .map(|(_, &idx)| idx)
            .collect();

        if seeds.is_empty() {
            return GraphRagResult::err(
                &format!("no nodes matching '{}'", target),
                start.elapsed().as_millis() as u64,
            );
        }

        let direction = match kind {
            QueryKind::Callers | QueryKind::Dependents => Direction::Incoming,
            _ => Direction::Outgoing,
        };

        let max_nodes = 50usize;
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, u32)> = VecDeque::new();
        let mut result_nodes: Vec<GraphNode> = Vec::new();
        let mut result_edges: Vec<GraphEdge> = Vec::new();

        // Seed the BFS
        for &seed in &seeds {
            if visited.insert(seed) {
                queue.push_back((seed, 0));
                let data = &self.graph[seed];
                result_nodes.push(GraphNode {
                    symbol: data.symbol.clone(),
                    file: data.file.clone(),
                    line: data.line,
                    symbol_kind: data.kind.to_string(),
                    depth: 0,
                });
            }
        }

        // BFS
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= hops || result_nodes.len() >= max_nodes {
                continue;
            }

            let neighbors: Vec<_> = self.graph.neighbors_directed(current, direction).collect();

            for neighbor in neighbors {
                if visited.insert(neighbor) && result_nodes.len() < max_nodes {
                    let data = &self.graph[neighbor];
                    let next_depth = depth + 1;
                    result_nodes.push(GraphNode {
                        symbol: data.symbol.clone(),
                        file: data.file.clone(),
                        line: data.line,
                        symbol_kind: data.kind.to_string(),
                        depth: next_depth,
                    });
                    queue.push_back((neighbor, next_depth));

                    // Record the edge
                    let current_data = &self.graph[current];
                    let (from, to) = match direction {
                        Direction::Outgoing => (current_data.symbol.clone(), data.symbol.clone()),
                        Direction::Incoming => (data.symbol.clone(), current_data.symbol.clone()),
                    };
                    // Find edge kind
                    let edge_label = match direction {
                        Direction::Outgoing => self
                            .graph
                            .find_edge(current, neighbor)
                            .map(|e| self.graph[e].to_string()),
                        Direction::Incoming => self
                            .graph
                            .find_edge(neighbor, current)
                            .map(|e| self.graph[e].to_string()),
                    };
                    result_edges.push(GraphEdge {
                        from,
                        to,
                        relation: edge_label.unwrap_or_else(|| "unknown".to_string()),
                    });
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;
        let total = result_nodes.len();
        let mut result = GraphRagResult::ok(result_nodes, result_edges, elapsed);
        result.truncate_to(max_nodes);
        if total > max_nodes {
            result.total_results = total;
        }
        result
    }

    /// Format subgraph context for prompt injection.
    pub fn to_dependency_section(&self, target_symbols: &[&str], hops: u32) -> String {
        use crate::reviewer_tools::graph_rag::QueryKind;

        let mut sections = Vec::new();

        for &target in target_symbols {
            let result = self.get_subgraph_context(target, hops, QueryKind::Callees);
            if result.is_success() && !result.nodes.is_empty() {
                let mut section = format!("## Dependencies of `{}`\n", target);
                for node in &result.nodes {
                    let indent = "  ".repeat(node.depth as usize);
                    section.push_str(&format!(
                        "{}[{}] {} ({}:{})\n",
                        indent, node.symbol_kind, node.symbol, node.file, node.line
                    ));
                }
                if !result.edges.is_empty() {
                    section.push_str("\nEdges:\n");
                    for edge in &result.edges {
                        section.push_str(&format!(
                            "  {} --{}-> {}\n",
                            edge.from, edge.relation, edge.to
                        ));
                    }
                }
                sections.push(section);
            }
        }

        if sections.is_empty() {
            "No dependency information found.".to_string()
        } else {
            sections.join("\n")
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reviewer_tools::graph_rag::QueryKind;

    #[test]
    fn test_calls_edges() {
        let source = r#"
fn callee() -> u32 {
    42
}

fn caller() -> u32 {
    callee()
}
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        assert!(graph.node_count() >= 2, "Should have at least 2 nodes");
        assert!(graph.edge_count() >= 1, "Should have at least 1 call edge");

        // BFS from caller should find callee
        let result = graph.get_subgraph_context("caller", 2, QueryKind::Callees);
        assert!(result.is_success());
        let symbols: Vec<&str> = result.nodes.iter().map(|n| n.symbol.as_str()).collect();
        assert!(symbols.contains(&"caller"), "Should include caller");
        assert!(symbols.contains(&"callee"), "Should include callee");
    }

    #[test]
    fn test_implements_edges() {
        let source = r#"
trait Greeter {
    fn greet(&self) -> String;
}

struct Bot {
    name: String,
}

impl Greeter for Bot {
    fn greet(&self) -> String {
        format!("Hello from {}", self.name)
    }
}
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        // Should have nodes for Greeter, Bot
        assert!(graph.node_count() >= 2);

        // Check for Implements edge: Bot -> Greeter
        let result = graph.get_subgraph_context("Bot", 2, QueryKind::Callees);
        assert!(result.is_success());

        // The edge list should contain an "implements" edge
        let has_impl_edge = result.edges.iter().any(|e| e.relation == "implements");
        assert!(
            has_impl_edge,
            "Should have an implements edge. Edges: {:?}",
            result.edges
        );
    }

    #[test]
    fn test_bfs_depth_limiting() {
        let source = r#"
fn d() -> u32 { 1 }
fn c() -> u32 { d() }
fn b() -> u32 { c() }
fn a() -> u32 { b() }
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        // With hops=1 from "a", should only reach "b" (direct callee)
        let result = graph.get_subgraph_context("::a", 1, QueryKind::Callees);
        assert!(result.is_success());
        // depth 0 = a, depth 1 = b
        let max_depth = result.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        assert!(
            max_depth <= 1,
            "Max depth should be <= 1, got {}",
            max_depth
        );

        // With hops=3 from "a", should reach all the way to "d"
        let result = graph.get_subgraph_context("::a", 3, QueryKind::Callees);
        assert!(result.is_success());
        let symbols: Vec<&str> = result.nodes.iter().map(|n| n.symbol.as_str()).collect();
        assert!(symbols.contains(&"a"));
        assert!(
            symbols.contains(&"d"),
            "Should reach d with 3 hops. Got: {:?}",
            symbols
        );
    }

    #[test]
    fn test_callers_direction() {
        let source = r#"
fn target() -> u32 { 42 }
fn caller_one() -> u32 { target() }
fn caller_two() -> u32 { target() }
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        // QueryKind::Callers should follow incoming edges
        let result = graph.get_subgraph_context("target", 1, QueryKind::Callers);
        assert!(result.is_success());

        let symbols: Vec<&str> = result.nodes.iter().map(|n| n.symbol.as_str()).collect();
        assert!(symbols.contains(&"target"));
        // At least one caller should be found
        let callers: Vec<&&str> = symbols.iter().filter(|s| s.contains("caller")).collect();
        assert!(
            !callers.is_empty(),
            "Should find at least one caller. Got: {:?}",
            symbols
        );
    }

    #[test]
    fn test_cross_file_calls() {
        let file_a = r#"
fn helper() -> u32 { 1 }
"#;
        let file_b = r#"
fn consumer() -> u32 { helper() }
"#;
        let graph = SemanticCodeGraph::from_sources(&[("a.rs", file_a), ("b.rs", file_b)]);

        let result = graph.get_subgraph_context("consumer", 2, QueryKind::Callees);
        assert!(result.is_success());

        let symbols: Vec<&str> = result.nodes.iter().map(|n| n.symbol.as_str()).collect();
        assert!(symbols.contains(&"consumer"));
        assert!(
            symbols.contains(&"helper"),
            "Should resolve cross-file call. Got: {:?}",
            symbols
        );
    }

    #[test]
    fn test_struct_instantiation() {
        let source = r#"
struct Config {
    name: String,
}

fn make_config() -> Config {
    Config { name: "test".to_string() }
}
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        let result = graph.get_subgraph_context("make_config", 2, QueryKind::Callees);
        assert!(result.is_success());

        let has_instantiates = result.edges.iter().any(|e| e.relation == "instantiates");
        assert!(
            has_instantiates,
            "Should have instantiates edge. Edges: {:?}",
            result.edges
        );
    }

    #[test]
    fn test_no_match_returns_error() {
        let source = "fn nothing() {}";
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        let result = graph.get_subgraph_context("nonexistent_symbol", 2, QueryKind::Callees);
        assert!(!result.is_success());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_empty_graph() {
        let graph = SemanticCodeGraph::from_sources(&[]);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_to_dependency_section() {
        let source = r#"
fn callee() -> u32 { 42 }
fn caller() -> u32 { callee() }
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);
        let section = graph.to_dependency_section(&["caller"], 2);
        assert!(section.contains("caller"), "Section should mention caller");
    }

    #[test]
    fn test_method_in_impl() {
        let source = r#"
struct Foo;

impl Foo {
    fn do_stuff(&self) -> u32 { 42 }
}

fn uses_foo() {
    let f = Foo;
}
"#;
        let graph = SemanticCodeGraph::from_sources(&[("test.rs", source)]);

        // Method should be registered as Foo::do_stuff
        let has_method = graph.index.keys().any(|k| k.contains("Foo::do_stuff"));
        assert!(
            has_method,
            "Should have Foo::do_stuff. Keys: {:?}",
            graph.index.keys().collect::<Vec<_>>()
        );
    }
}
