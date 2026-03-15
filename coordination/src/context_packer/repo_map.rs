//! RepoMap — Whole-codebase structural map via tree-sitter AST parsing.
//!
//! Generates a token-budgeted map of the entire codebase showing public
//! symbols (structs, traits, functions, impls) organized by file. Inspired
//! by aider's RepoMap which uses tree-sitter + PageRank to give the agent
//! a complete mental model of the codebase without reading every file.
//!
//! The map is injected into the task prompt so the manager/planner knows
//! WHERE things are before making any tool calls — eliminating the
//! "100+ blind file reads" problem observed in dogfood runs.
//!
//! # Token Budget
//!
//! The map targets ~2000 tokens (8000 chars). Files are ranked by:
//! 1. Import frequency (how many other files `use` this module)
//! 2. Public symbol count (more symbols = more important)
//! 3. Relevance to the issue objective (keyword matching)
//!
//! Low-ranked files are elided to just their path; high-ranked files
//! show full public symbol signatures.

use std::collections::HashMap;
use std::path::Path;

use super::ast_index::{FileSymbolIndex, SymbolKind};
use super::file_walker::FileWalker;

/// Maximum character budget for the repo map (~2000 tokens at 4 chars/token).
const DEFAULT_CHAR_BUDGET: usize = 8000;

/// A scored file entry in the repo map.
#[derive(Debug)]
struct ScoredFile {
    /// Relative path from worktree root
    path: String,
    /// Symbol index for this file
    index: FileSymbolIndex,
    /// Importance score (higher = show more detail)
    score: f64,
}

/// Generate a repo map for the given worktree.
///
/// Returns a formatted string showing the codebase structure organized by
/// directory, with public symbols listed for the most important files.
///
/// # Arguments
/// * `worktree_root` — Path to the worktree to map
/// * `objective` — Issue objective text (used for keyword relevance scoring)
/// * `char_budget` — Maximum characters for the map (0 = default 8000)
pub fn generate_repo_map(
    worktree_root: &Path,
    objective: &str,
    char_budget: usize,
) -> String {
    let budget = if char_budget == 0 {
        DEFAULT_CHAR_BUDGET
    } else {
        char_budget
    };

    let walker = FileWalker::new(worktree_root);
    let rust_files = walker.rust_files();

    if rust_files.is_empty() {
        return String::new();
    }

    // Parse all files and compute scores
    let mut scored_files = Vec::new();
    let mut import_counts: HashMap<String, usize> = HashMap::new();

    // First pass: parse all files and count imports
    let mut all_sources: Vec<(String, String)> = Vec::new();
    for file_path in &rust_files {
        let rel_path = file_path
            .strip_prefix(worktree_root)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();

        // Skip test files, build artifacts, and vendored code
        if rel_path.contains("/target/")
            || rel_path.contains("/tests/")
            || rel_path.contains("patches/")
        {
            continue;
        }

        if let Ok(source) = std::fs::read_to_string(file_path) {
            // Count imports: how many times is this module referenced?
            // Extract module path from file path for import matching
            let module_stem = module_path_from_file(&rel_path);
            all_sources.push((rel_path, source));
            if !module_stem.is_empty() {
                import_counts.entry(module_stem).or_insert(0);
            }
        }
    }

    // Count cross-file references
    for (_, source) in &all_sources {
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("use crate::") || trimmed.starts_with("use super::") {
                // Extract the first path segment after crate::/super::
                let after = if let Some(rest) = trimmed.strip_prefix("use crate::") {
                    rest
                } else if let Some(rest) = trimmed.strip_prefix("use super::") {
                    rest
                } else {
                    continue;
                };

                // Get the module name (first segment before :: or {)
                let module = after
                    .split([':', '{', ';', ' '])
                    .next()
                    .unwrap_or("");

                if !module.is_empty() {
                    *import_counts.entry(module.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    // Second pass: build symbol indices and score files
    let objective_lower = objective.to_lowercase();
    let objective_words: Vec<&str> = objective_lower
        .split_whitespace()
        .filter(|w| w.len() > 3) // Skip short words
        .collect();

    for (rel_path, source) in &all_sources {
        let index = FileSymbolIndex::from_source(rel_path, source);

        let module_stem = module_path_from_file(rel_path);
        let import_score = *import_counts.get(&module_stem).unwrap_or(&0) as f64;

        // Public symbol count score
        let pub_count = index
            .symbols
            .iter()
            .filter(|s| s.is_public)
            .count() as f64;

        // Keyword relevance score — does the file path or any symbol name
        // match words from the issue objective?
        let path_lower = rel_path.to_lowercase();
        let mut keyword_score: f64 = 0.0;
        for word in &objective_words {
            if path_lower.contains(word) {
                keyword_score += 3.0;
            }
            for sym in &index.symbols {
                let name_lower = sym.name.to_lowercase();
                if name_lower.contains(word) {
                    keyword_score += 1.0;
                }
            }
        }

        // Composite score: imports weighted highest, then keywords, then symbol count
        let score = import_score * 3.0 + keyword_score * 2.0 + pub_count;

        scored_files.push(ScoredFile {
            path: rel_path.clone(),
            index,
            score,
        });
    }

    // Sort by score descending
    scored_files.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Build the map within budget
    format_repo_map(&scored_files, budget)
}

/// Format the scored files into a structured map string.
fn format_repo_map(scored_files: &[ScoredFile], char_budget: usize) -> String {
    let mut output = String::new();
    let mut chars_used = 0;

    // Group files by top-level directory
    let mut by_dir: Vec<(&str, Vec<&ScoredFile>)> = Vec::new();
    let mut dir_map: HashMap<&str, usize> = HashMap::new();

    for sf in scored_files {
        let dir = sf
            .path
            .find('/')
            .map(|i| &sf.path[..i])
            .unwrap_or(".");

        if let Some(&idx) = dir_map.get(dir) {
            by_dir[idx].1.push(sf);
        } else {
            dir_map.insert(dir, by_dir.len());
            by_dir.push((dir, vec![sf]));
        }
    }

    // Sort directories: coordination first, then crates/swarm-agents, then others
    by_dir.sort_by_key(|(dir, _)| match *dir {
        "coordination" => 0,
        "crates" => 1,
        _ => 2,
    });

    for (dir, files) in &by_dir {
        if chars_used >= char_budget {
            break;
        }

        let dir_header = format!("\n### {dir}/\n");
        output.push_str(&dir_header);
        chars_used += dir_header.len();

        for sf in files {
            if chars_used >= char_budget {
                break;
            }

            let pub_symbols: Vec<&super::ast_index::RustSymbol> = sf
                .index
                .symbols
                .iter()
                .filter(|s| s.is_public || s.kind == SymbolKind::Impl)
                .collect();

            if pub_symbols.is_empty() {
                // No public symbols — just list the file path
                let line = format!("  {}\n", sf.path);
                if chars_used + line.len() > char_budget {
                    break;
                }
                output.push_str(&line);
                chars_used += line.len();
                continue;
            }

            // Show file path and its public symbols
            let file_header = format!("  {} ({})\n", sf.path, pub_symbols.len());
            if chars_used + file_header.len() > char_budget {
                break;
            }
            output.push_str(&file_header);
            chars_used += file_header.len();

            // For high-scored files (top 20), show symbol signatures
            // For lower-scored files, just show the count
            if sf.score >= 3.0 {
                // Sort: types first, then functions, then impls
                let mut sorted_syms = pub_symbols.clone();
                sorted_syms.sort_by_key(|s| match s.kind {
                    SymbolKind::Struct | SymbolKind::Enum | SymbolKind::Trait => 0,
                    SymbolKind::TypeAlias | SymbolKind::Const => 1,
                    SymbolKind::Function => 2,
                    SymbolKind::Impl => 3,
                    _ => 4,
                });

                for sym in sorted_syms {
                    let vis = if sym.is_public { "pub " } else { "" };
                    let sig_line = format!(
                        "    {}{} {}  (L{})\n",
                        vis,
                        sym.kind,
                        sym.name,
                        sym.start_line + 1,
                    );
                    if chars_used + sig_line.len() > char_budget {
                        output.push_str("    ⋮...\n");
                        chars_used += 10;
                        break;
                    }
                    output.push_str(&sig_line);
                    chars_used += sig_line.len();
                }
            }
        }
    }

    output
}

/// Convert a file path like `crates/swarm-agents/src/tools/bundles.rs`
/// to a module path like `bundles` (last stem).
fn module_path_from_file(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // mod.rs → use parent directory name
    if stem == "mod" || stem == "lib" || stem == "main" {
        Path::new(path)
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_generate_repo_map_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let map = generate_repo_map(dir.path(), "Fix a bug", 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_generate_repo_map_basic() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(
            src.join("lib.rs"),
            r#"
pub struct Config {
    pub name: String,
}

pub fn process(config: &Config) -> Result<(), String> {
    Ok(())
}

pub trait Handler {
    fn handle(&self);
}
"#,
        )
        .unwrap();

        fs::write(
            src.join("utils.rs"),
            r#"
use crate::Config;

pub fn helper() -> bool {
    true
}
"#,
        )
        .unwrap();

        let map = generate_repo_map(dir.path(), "Fix Config processing", 0);

        assert!(!map.is_empty());
        // lib.rs should appear (has Config which matches objective)
        assert!(map.contains("lib.rs"), "Map should include lib.rs: {map}");
        // Config struct should appear (keyword match)
        assert!(
            map.contains("Config"),
            "Map should include Config symbol: {map}"
        );
    }

    #[test]
    fn test_generate_repo_map_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // Create many files
        for i in 0..20 {
            fs::write(
                src.join(format!("module_{i}.rs")),
                format!(
                    "pub struct Type{i} {{}}\npub fn func_{i}() {{}}\npub trait Trait{i} {{}}\n"
                ),
            )
            .unwrap();
        }

        // Small budget — should truncate
        let map = generate_repo_map(dir.path(), "test", 500);
        assert!(map.len() <= 600); // Allow some slack for final line
    }

    #[test]
    fn test_module_path_from_file() {
        assert_eq!(module_path_from_file("src/tools/bundles.rs"), "bundles");
        assert_eq!(
            module_path_from_file("crates/swarm-agents/src/orchestrator/mod.rs"),
            "orchestrator"
        );
        assert_eq!(module_path_from_file("src/lib.rs"), "src");
        assert_eq!(module_path_from_file("main.rs"), "");
    }

    #[test]
    fn test_import_scoring() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // A module that gets imported by many others should score higher
        fs::write(
            src.join("core.rs"),
            "pub struct CoreType {}\npub fn core_fn() {}\n",
        )
        .unwrap();

        fs::write(src.join("a.rs"), "use crate::core;\npub fn a() {}\n").unwrap();
        fs::write(src.join("b.rs"), "use crate::core;\npub fn b() {}\n").unwrap();
        fs::write(src.join("c.rs"), "use crate::core;\npub fn c() {}\n").unwrap();
        fs::write(src.join("lonely.rs"), "pub fn lonely() {}\n").unwrap();

        let map = generate_repo_map(dir.path(), "something", 0);

        // core.rs should appear before lonely.rs (more imports)
        let core_pos = map.find("core.rs").unwrap_or(usize::MAX);
        let lonely_pos = map.find("lonely.rs").unwrap_or(usize::MAX);
        assert!(
            core_pos < lonely_pos,
            "core.rs (imported 3x) should rank above lonely.rs: {map}"
        );
    }
}
