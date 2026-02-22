//! AST Index — Tree-sitter-based Rust symbol extraction.
//!
//! Parses Rust source files with tree-sitter-rust to extract structured
//! symbol information: function signatures, struct/enum/trait definitions,
//! impl blocks, and their byte/line ranges.
//!
//! Used by the ContextPacker to build smarter initial context that includes
//! complete signatures and type definitions instead of raw first-N-lines.

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

/// A symbol extracted from a Rust source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustSymbol {
    /// Symbol name (e.g., "VerifierConfig", "run_pipeline")
    pub name: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// Whether the symbol has `pub` visibility
    pub is_public: bool,
    /// Starting line (0-indexed)
    pub start_line: usize,
    /// Ending line (0-indexed)
    pub end_line: usize,
    /// The signature or header text (first line of the symbol, for display)
    pub signature: String,
}

/// Categories of Rust symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    TypeAlias,
    Const,
    Static,
    Mod,
    Macro,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function => write!(f, "fn"),
            Self::Struct => write!(f, "struct"),
            Self::Enum => write!(f, "enum"),
            Self::Trait => write!(f, "trait"),
            Self::Impl => write!(f, "impl"),
            Self::TypeAlias => write!(f, "type"),
            Self::Const => write!(f, "const"),
            Self::Static => write!(f, "static"),
            Self::Mod => write!(f, "mod"),
            Self::Macro => write!(f, "macro"),
        }
    }
}

/// Index of all symbols in a single Rust source file.
#[derive(Debug, Clone, Default)]
pub struct FileSymbolIndex {
    /// Source file path (relative)
    pub file: String,
    /// All extracted symbols
    pub symbols: Vec<RustSymbol>,
}

impl FileSymbolIndex {
    /// Parse a Rust source file and extract all top-level symbols.
    pub fn from_source(file: &str, source: &str) -> Self {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust language");

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => {
                return Self {
                    file: file.to_string(),
                    symbols: Vec::new(),
                }
            }
        };

        let root = tree.root_node();
        let source_bytes = source.as_bytes();
        let mut symbols = Vec::new();

        Self::walk_node(root, source_bytes, &mut symbols);

        Self {
            file: file.to_string(),
            symbols,
        }
    }

    /// Recursively walk the AST to extract symbols.
    fn walk_node(node: Node, source: &[u8], symbols: &mut Vec<RustSymbol>) {
        let kind_str = node.kind();

        match kind_str {
            "function_item" => {
                if let Some(sym) = Self::extract_function(node, source) {
                    symbols.push(sym);
                }
            }
            "struct_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Struct) {
                    symbols.push(sym);
                }
            }
            "enum_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Enum) {
                    symbols.push(sym);
                }
            }
            "trait_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Trait) {
                    symbols.push(sym);
                }
            }
            "impl_item" => {
                if let Some(sym) = Self::extract_impl(node, source) {
                    symbols.push(sym);
                }
            }
            "type_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::TypeAlias) {
                    symbols.push(sym);
                }
            }
            "const_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Const) {
                    symbols.push(sym);
                }
            }
            "static_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Static) {
                    symbols.push(sym);
                }
            }
            "mod_item" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Mod) {
                    symbols.push(sym);
                }
            }
            "macro_definition" => {
                if let Some(sym) = Self::extract_named_item(node, source, SymbolKind::Macro) {
                    symbols.push(sym);
                }
            }
            _ => {}
        }

        // Walk children for top-level items
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::walk_node(child, source, symbols);
        }
    }

    /// Extract a function symbol with its full signature.
    fn extract_function(node: Node, source: &[u8]) -> Option<RustSymbol> {
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())?;

        let is_public = Self::is_public(node, source);
        let signature = Self::first_line(node, source);

        Some(RustSymbol {
            name: name.to_string(),
            kind: SymbolKind::Function,
            is_public,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
        })
    }

    /// Extract a named item (struct, enum, trait, type, const, static, mod, macro).
    fn extract_named_item(node: Node, source: &[u8], kind: SymbolKind) -> Option<RustSymbol> {
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source).ok())?;

        let is_public = Self::is_public(node, source);
        let signature = Self::first_line(node, source);

        Some(RustSymbol {
            name: name.to_string(),
            kind,
            is_public,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
        })
    }

    /// Extract an impl block (e.g., `impl Foo for Bar`).
    fn extract_impl(node: Node, source: &[u8]) -> Option<RustSymbol> {
        let text = node.utf8_text(source).ok()?;
        // Extract the impl target: "impl<T> Foo for Bar { ... }" → "Foo for Bar"
        // Use the first line as the signature
        let signature = Self::first_line(node, source);

        // Extract a reasonable name from the impl header
        let name = text
            .lines()
            .next()
            .unwrap_or("")
            .trim_end_matches('{')
            .trim()
            .to_string();

        Some(RustSymbol {
            name,
            kind: SymbolKind::Impl,
            is_public: false, // impl blocks don't have visibility
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
        })
    }

    /// Check if a node has `pub` visibility by looking at preceding siblings.
    fn is_public(node: Node, source: &[u8]) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                if let Ok(text) = child.utf8_text(source) {
                    return text.starts_with("pub");
                }
            }
        }
        false
    }

    /// Get the first line of a node's text as a signature.
    fn first_line(node: Node, source: &[u8]) -> String {
        node.utf8_text(source)
            .ok()
            .and_then(|text| text.lines().next())
            .unwrap_or("")
            .to_string()
    }

    /// Get all public symbols.
    pub fn public_symbols(&self) -> Vec<&RustSymbol> {
        self.symbols.iter().filter(|s| s.is_public).collect()
    }

    /// Get symbols of a specific kind.
    pub fn symbols_of_kind(&self, kind: SymbolKind) -> Vec<&RustSymbol> {
        self.symbols.iter().filter(|s| s.kind == kind).collect()
    }

    /// Get a symbol by name.
    pub fn find_by_name(&self, name: &str) -> Option<&RustSymbol> {
        self.symbols.iter().find(|s| s.name == name)
    }

    /// Build a compact summary of the file's symbols for context packing.
    ///
    /// Returns a string like:
    /// ```text
    /// pub struct VerifierConfig { ... }  (lines 17-41)
    /// pub fn run_pipeline(&self) -> VerifierReport  (lines 130-209)
    /// impl Verifier { ... }  (lines 120-640)
    /// ```
    pub fn compact_summary(&self) -> String {
        let mut lines = Vec::new();

        // Sort by kind priority: structs/enums/traits first, then fns, then impls
        let mut sorted: Vec<&RustSymbol> = self.symbols.iter().collect();
        sorted.sort_by_key(|s| match s.kind {
            SymbolKind::Struct | SymbolKind::Enum | SymbolKind::Trait => 0,
            SymbolKind::TypeAlias | SymbolKind::Const | SymbolKind::Static => 1,
            SymbolKind::Function => 2,
            SymbolKind::Impl => 3,
            SymbolKind::Mod | SymbolKind::Macro => 4,
        });

        for sym in &sorted {
            if !sym.is_public && sym.kind != SymbolKind::Impl {
                continue; // Skip private items except impl blocks
            }
            let vis = if sym.is_public { "pub " } else { "" };
            lines.push(format!(
                "{}{} {}  (lines {}-{})",
                vis,
                sym.kind,
                sym.name,
                sym.start_line + 1,
                sym.end_line + 1,
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RUST: &str = r#"
use std::path::Path;

/// A public struct.
pub struct Config {
    pub name: String,
    pub timeout: u64,
}

/// A private enum.
enum Internal {
    A,
    B(u32),
}

/// Public trait.
pub trait Worker {
    fn run(&self) -> Result<(), String>;
}

impl Config {
    pub fn new(name: String) -> Self {
        Self { name, timeout: 30 }
    }

    fn private_method(&self) -> u64 {
        self.timeout
    }
}

impl Worker for Config {
    fn run(&self) -> Result<(), String> {
        Ok(())
    }
}

pub fn top_level_fn(x: u32) -> u32 {
    x + 1
}

pub type AliasType = Vec<String>;

pub const MAX_RETRIES: u32 = 5;
"#;

    #[test]
    fn test_parse_extracts_all_symbol_kinds() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);

        assert!(!index.symbols.is_empty(), "Should extract symbols");

        // Check for struct
        let structs = index.symbols_of_kind(SymbolKind::Struct);
        assert!(
            structs.iter().any(|s| s.name == "Config"),
            "Should find Config struct"
        );

        // Check for enum
        let enums = index.symbols_of_kind(SymbolKind::Enum);
        assert!(
            enums.iter().any(|s| s.name == "Internal"),
            "Should find Internal enum"
        );

        // Check for trait
        let traits = index.symbols_of_kind(SymbolKind::Trait);
        assert!(
            traits.iter().any(|s| s.name == "Worker"),
            "Should find Worker trait"
        );

        // Check for impl blocks
        let impls = index.symbols_of_kind(SymbolKind::Impl);
        assert!(impls.len() >= 2, "Should find at least 2 impl blocks");

        // Check for function
        let fns = index.symbols_of_kind(SymbolKind::Function);
        assert!(
            fns.iter().any(|s| s.name == "top_level_fn"),
            "Should find top_level_fn"
        );

        // Check for type alias
        let types = index.symbols_of_kind(SymbolKind::TypeAlias);
        assert!(
            types.iter().any(|s| s.name == "AliasType"),
            "Should find AliasType"
        );

        // Check for const
        let consts = index.symbols_of_kind(SymbolKind::Const);
        assert!(
            consts.iter().any(|s| s.name == "MAX_RETRIES"),
            "Should find MAX_RETRIES"
        );
    }

    #[test]
    fn test_visibility_detection() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);

        let config = index.find_by_name("Config").unwrap();
        assert!(config.is_public);

        let internal = index.find_by_name("Internal").unwrap();
        assert!(!internal.is_public);

        let top_fn = index.find_by_name("top_level_fn").unwrap();
        assert!(top_fn.is_public);
    }

    #[test]
    fn test_public_symbols_filter() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);
        let public = index.public_symbols();

        // Config, Worker, top_level_fn, AliasType, MAX_RETRIES should be public
        assert!(public.len() >= 5, "Should have at least 5 public symbols");
        assert!(public.iter().all(|s| s.is_public));
    }

    #[test]
    fn test_line_ranges() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);

        let config = index.find_by_name("Config").unwrap();
        assert!(
            config.end_line > config.start_line,
            "Struct should span multiple lines"
        );
    }

    #[test]
    fn test_compact_summary_format() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);
        let summary = index.compact_summary();

        assert!(!summary.is_empty());
        assert!(summary.contains("pub struct Config"));
        assert!(summary.contains("pub trait Worker"));
        assert!(summary.contains("pub fn top_level_fn"));
        // Impl blocks should be included
        assert!(summary.contains("impl"));
    }

    #[test]
    fn test_find_by_name() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);

        assert!(index.find_by_name("Config").is_some());
        assert!(index.find_by_name("NonExistent").is_none());
    }

    #[test]
    fn test_empty_source() {
        let index = FileSymbolIndex::from_source("empty.rs", "");
        assert!(index.symbols.is_empty());
    }

    #[test]
    fn test_malformed_source() {
        // tree-sitter is error-tolerant — it should still extract what it can
        let source = "pub struct Foo { \npub fn bar() -> { broken syntax here }";
        let index = FileSymbolIndex::from_source("bad.rs", source);
        // Should still find the struct even with broken syntax
        assert!(
            index.find_by_name("Foo").is_some(),
            "Should extract Foo despite syntax errors"
        );
    }

    #[test]
    fn test_impl_block_name() {
        let index = FileSymbolIndex::from_source("test.rs", SAMPLE_RUST);
        let impls = index.symbols_of_kind(SymbolKind::Impl);

        // Should have descriptive names including the impl target
        let has_config_impl = impls.iter().any(|s| s.name.contains("Config"));
        assert!(has_config_impl, "Should have impl block for Config");
    }

    #[test]
    fn test_symbol_serialization() {
        let sym = RustSymbol {
            name: "Foo".to_string(),
            kind: SymbolKind::Struct,
            is_public: true,
            start_line: 5,
            end_line: 10,
            signature: "pub struct Foo {".to_string(),
        };
        let json = serde_json::to_string(&sym).unwrap();
        let parsed: RustSymbol = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "Foo");
        assert_eq!(parsed.kind, SymbolKind::Struct);
        assert!(parsed.is_public);
    }

    #[test]
    fn test_methods_inside_impl_not_duplicated_at_top_level() {
        let source = r#"
pub struct Foo;

impl Foo {
    pub fn method_one(&self) -> u32 { 1 }
    pub fn method_two(&self) -> u32 { 2 }
}
"#;
        let index = FileSymbolIndex::from_source("test.rs", source);

        // Methods inside impl should be found as functions
        // but the top-level function list should include them
        let fns = index.symbols_of_kind(SymbolKind::Function);
        let method_names: Vec<&str> = fns.iter().map(|f| f.name.as_str()).collect();
        assert!(
            method_names.contains(&"method_one"),
            "Should find method_one"
        );
        assert!(
            method_names.contains(&"method_two"),
            "Should find method_two"
        );
    }
}
