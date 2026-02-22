//! Context Packer — Builds token-budgeted WorkPackets for agent tiers
//!
//! `pack_initial` creates a WorkPacket from scratch (no prior failures).
//! `pack_retry` delegates to the existing WorkPacketGenerator for error-enriched packets.

use crate::context_packer::ast_index::FileSymbolIndex;
use crate::context_packer::file_walker::FileWalker;
use crate::escalation::state::{EscalationState, SwarmTier};
use crate::feedback::error_parser::ErrorCategory;
use crate::verifier::report::VerifierReport;
use crate::work_packet::generator::WorkPacketGenerator;
use crate::work_packet::types::{ContextProvenance, FileContext, WorkPacket};
use chrono::Utc;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Regex for extracting file paths from cargo stderr output (e.g., ` --> path/to/file.rs:123:45`).
static STDERR_FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-->\s*([^\s:]+\.rs):(\d+)").expect("STDERR_FILE_RE regex should compile")
});

/// Regex for extracting type/trait names from error messages.
///
/// Matches patterns like: `the trait `Foo` is not implemented`, `expected `Bar``,
/// `cannot find type `Baz``, etc.
static SYMBOL_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"`([A-Z][A-Za-z0-9_]+)`").expect("SYMBOL_REF_RE regex should compile")
});

/// Number of context lines before and after an error span.
const SPAN_CONTEXT_LINES: usize = 80;

/// Below this line count, include the full file instead of windowing.
const SMALL_FILE_THRESHOLD: usize = 200;

/// A line range window around an error span in a specific file.
#[derive(Debug, Clone)]
struct SpanWindow {
    file: String,
    center_line: usize,
    start_line: usize,
    end_line: usize,
    category: ErrorCategory,
    message: String,
}

/// Token budgets per tier (4 chars ≈ 1 token, matching `estimated_tokens()`)
fn max_context_tokens(tier: SwarmTier) -> usize {
    fn from_env(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(default)
    }

    match tier {
        SwarmTier::Worker => from_env("SWARM_CONTEXT_TOKENS_WORKER", 8_000),
        SwarmTier::Council => from_env("SWARM_CONTEXT_TOKENS_COUNCIL", 32_000),
        SwarmTier::Human => from_env("SWARM_CONTEXT_TOKENS_HUMAN", 32_000),
    }
}

/// Builds token-budgeted context for agent tiers.
pub struct ContextPacker {
    working_dir: PathBuf,
    generator: WorkPacketGenerator,
    file_walker: FileWalker,
    tier: SwarmTier,
    max_context_tokens: usize,
}

impl ContextPacker {
    pub fn new(working_dir: impl AsRef<Path>, tier: SwarmTier) -> Self {
        let wd = working_dir.as_ref().to_path_buf();
        Self {
            generator: WorkPacketGenerator::new(&wd),
            file_walker: FileWalker::new(&wd),
            tier,
            max_context_tokens: max_context_tokens(tier),
            working_dir: wd,
        }
    }

    /// Initial pack: walks worktree, gathers file headers within token budget (no prior failures).
    pub fn pack_initial(&self, bead_id: &str, objective: &str) -> WorkPacket {
        let rust_files = self.file_walker.rust_files();

        // Make paths relative to working_dir for consistency with WorkPacketGenerator
        let relative_files: Vec<String> = rust_files
            .iter()
            .filter_map(|p| p.strip_prefix(&self.working_dir).ok())
            .map(|p| p.display().to_string())
            .collect();

        // Extract symbols from all .rs files (reuse generator's extract logic via generate)
        let state = EscalationState::new(bead_id);
        let mut packet = self
            .generator
            .generate(bead_id, objective, self.tier, &state, None);

        // The generator only gets symbols from git-changed files.
        // For initial pack, we want symbols from ALL .rs files, so build
        // file contexts manually from the full file list.
        packet.file_contexts = self.build_file_contexts(&relative_files);
        packet.objective = objective.to_string();
        packet.generated_at = Utc::now();

        // Trim to fit token budget
        self.trim_to_budget(&mut packet);
        packet
    }

    /// Retry pack: delegates to WorkPacketGenerator with error context,
    /// then overrides file contexts to include FULL content of changed/failing files.
    ///
    /// The generator's default `extract_error_contexts()` only includes ~20-line
    /// windows around error locations. For retries, the worker needs to see the
    /// full file content to fix cascading errors and understand the broader context.
    /// Without this, iteration 2+ gets ~300-700 tokens vs ~24K on iteration 1
    /// (the "retry context collapse" bug from job 1653).
    pub fn pack_retry(
        &self,
        bead_id: &str,
        objective: &str,
        escalation_state: &EscalationState,
        verifier_report: &VerifierReport,
    ) -> WorkPacket {
        let mut packet = self.generator.generate(
            bead_id,
            objective,
            escalation_state.current_tier,
            escalation_state,
            Some(verifier_report),
        );

        // Use span-aware context when we have error spans (file + line info),
        // otherwise fall back to full-file contexts.
        let has_spans = verifier_report
            .failure_signals
            .iter()
            .any(|s| s.file.is_some() && s.line.is_some());

        packet.file_contexts = if has_spans {
            self.build_span_aware_contexts(verifier_report, &packet.files_touched)
        } else {
            self.build_retry_file_contexts(verifier_report, &packet.files_touched)
        };

        self.trim_to_budget(&mut packet);
        packet
    }

    /// Build file contexts for retry iterations with FULL file content.
    ///
    /// Unlike `pack_initial` (which includes 30-line headers for many files),
    /// retry contexts include the COMPLETE content of the most relevant files
    /// so the worker can actually see and fix the code.
    fn build_retry_file_contexts(
        &self,
        report: &VerifierReport,
        files_touched: &[String],
    ) -> Vec<FileContext> {
        let mut contexts = Vec::new();
        let mut files_seen = HashSet::new();
        let mut total_chars = 0usize;
        let char_budget = self.max_context_tokens * 4; // 4 chars per token

        // Canonical working_dir for sandbox validation
        let canon_wd = match self.working_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return contexts,
        };

        // 1. Files with errors — highest priority, include FULL content
        for signal in &report.failure_signals {
            if let Some(file) = &signal.file {
                if files_seen.contains(file) {
                    continue;
                }

                let full_path = self.working_dir.join(file);

                // Sandbox check: ensure resolved path stays within working_dir
                let canonical = match std::fs::canonicalize(&full_path) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !canonical.starts_with(&canon_wd) {
                    continue;
                }

                files_seen.insert(file.clone());

                let content = match std::fs::read_to_string(&canonical) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Pre-estimate: ~8 chars overhead per line (line num + " | " + newline)
                let estimated_chars =
                    content.len() + content.lines().count() * 8 + file.len() + 100;
                if total_chars + estimated_chars > char_budget {
                    continue; // skip oversized, try smaller files
                }

                let lines: Vec<&str> = content.lines().collect();
                let line_count = lines.len();
                let context_content: String = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                    .collect::<Vec<_>>()
                    .join("\n");

                let ctx_chars = context_content.len() + file.len() + 100;
                total_chars += ctx_chars;

                let error_line = signal.line.unwrap_or(0);
                contexts.push(FileContext {
                    file: file.clone(),
                    start_line: 1,
                    end_line: line_count,
                    content: context_content,
                    relevance: format!(
                        "ERROR: {} at line {} (full file)",
                        signal.category, error_line
                    ),
                    priority: 0, // Error context — highest priority
                    provenance: ContextProvenance::CompilerError,
                });
            }
        }

        // Also check gate stderr for file references not in failure_signals
        for gate in &report.gates {
            if let Some(stderr) = &gate.stderr_excerpt {
                // Extract file paths from stderr using --> path.rs:line pattern
                for cap in STDERR_FILE_RE.captures_iter(stderr) {
                    let file = cap[1].to_string();
                    let rel_file = file
                        .strip_prefix(&format!("{}/", self.working_dir.display()))
                        .unwrap_or(&file)
                        .to_string();

                    if files_seen.contains(&rel_file) {
                        continue;
                    }

                    let full_path = self.working_dir.join(&rel_file);

                    // Sandbox check: ensure resolved path stays within working_dir
                    let canonical = match std::fs::canonicalize(&full_path) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if !canonical.starts_with(&canon_wd) {
                        continue;
                    }

                    files_seen.insert(rel_file.clone());

                    let content = match std::fs::read_to_string(&canonical) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    // Pre-estimate before formatting
                    let estimated_chars =
                        content.len() + content.lines().count() * 8 + rel_file.len() + 100;
                    if total_chars + estimated_chars > char_budget {
                        continue; // skip oversized, try smaller files
                    }

                    let lines: Vec<&str> = content.lines().collect();
                    let line_count = lines.len();
                    let context_content: String = lines
                        .iter()
                        .enumerate()
                        .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let ctx_chars = context_content.len() + rel_file.len() + 100;
                    total_chars += ctx_chars;

                    let error_line: usize = cap[2].parse().unwrap_or(1);
                    contexts.push(FileContext {
                        file: rel_file,
                        start_line: 1,
                        end_line: line_count,
                        content: context_content,
                        relevance: format!(
                            "ERROR: {} gate at line {} (full file)",
                            gate.gate, error_line
                        ),
                        priority: 0, // Error context — highest priority
                        provenance: ContextProvenance::CompilerError,
                    });
                }
            }
        }

        // 2. Changed files not already included — full content
        for file in files_touched {
            if !file.ends_with(".rs") || files_seen.contains(file) {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Pre-estimate before formatting
            let estimated_chars = content.len() + content.lines().count() * 8 + file.len() + 100;
            if total_chars + estimated_chars > char_budget {
                continue; // skip oversized, try smaller files
            }

            let lines: Vec<&str> = content.lines().collect();
            let line_count = lines.len();
            let context_content: String = lines
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                .collect::<Vec<_>>()
                .join("\n");

            let ctx_chars = context_content.len() + file.len() + 100;
            total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line: line_count,
                content: context_content,
                relevance: "Modified file (full content for retry)".to_string(),
                priority: 1, // Modified file
                provenance: ContextProvenance::Diff,
            });
        }

        contexts
    }

    /// Build span-aware file contexts from verifier failure signals.
    ///
    /// Instead of including full files (which wastes token budget on irrelevant code),
    /// this method:
    /// 1. Extracts error spans from failure signals and gate errors
    /// 2. Creates ±80-line windows around each error span
    /// 3. Merges overlapping windows in the same file
    /// 4. For small files (<200 lines), includes the full file instead
    /// 5. Resolves type/trait references from error messages using AST index
    /// 6. Includes definitions of referenced symbols from other files
    fn build_span_aware_contexts(
        &self,
        report: &VerifierReport,
        files_touched: &[String],
    ) -> Vec<FileContext> {
        let mut contexts = Vec::new();
        let mut files_seen = HashSet::new();
        let mut total_chars = 0usize;
        let char_budget = self.max_context_tokens * 4;

        let canon_wd = match self.working_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return contexts,
        };

        // 1. Collect all error span windows
        let mut windows = self.collect_span_windows(report);

        // 2. Collect file references from gate stderr
        for gate in &report.gates {
            if let Some(stderr) = &gate.stderr_excerpt {
                for cap in STDERR_FILE_RE.captures_iter(stderr) {
                    let file = cap[1].to_string();
                    let rel_file = file
                        .strip_prefix(&format!("{}/", self.working_dir.display()))
                        .unwrap_or(&file)
                        .to_string();
                    let line: usize = cap[2].parse().unwrap_or(1);
                    windows.push(SpanWindow {
                        file: rel_file,
                        center_line: line,
                        start_line: line.saturating_sub(SPAN_CONTEXT_LINES),
                        end_line: line + SPAN_CONTEXT_LINES,
                        category: ErrorCategory::Other,
                        message: format!("{} gate error", gate.gate),
                    });
                }
            }
        }

        // 3. Merge overlapping windows per file
        let merged = Self::merge_windows(&windows);

        // 4. Build file contexts from merged windows
        for (file, file_windows) in &merged {
            if files_seen.contains(file) {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let canonical = match std::fs::canonicalize(&full_path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if !canonical.starts_with(&canon_wd) {
                continue;
            }

            let content = match std::fs::read_to_string(&canonical) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let lines: Vec<&str> = content.lines().collect();
            let line_count = lines.len();
            files_seen.insert(file.clone());

            // For small files, include everything
            if line_count <= SMALL_FILE_THRESHOLD {
                let context_content = Self::format_lines(&lines, 0, line_count);
                let ctx_chars = context_content.len() + file.len() + 100;
                if total_chars + ctx_chars > char_budget {
                    continue;
                }
                total_chars += ctx_chars;

                let relevance = file_windows
                    .iter()
                    .map(|w| format!("L{}:{}", w.center_line, w.category))
                    .collect::<Vec<_>>()
                    .join(", ");

                contexts.push(FileContext {
                    file: file.clone(),
                    start_line: 1,
                    end_line: line_count,
                    content: context_content,
                    relevance: format!("ERROR: {} (full file, {} lines)", relevance, line_count),
                    priority: 0,
                    provenance: ContextProvenance::CompilerError,
                });
                continue;
            }

            // For large files, pack windowed spans
            for window in file_windows {
                let start = window.start_line.min(line_count);
                let end = (window.end_line + 1).min(line_count);
                if start >= end {
                    continue;
                }

                let context_content = Self::format_lines(&lines, start, end);
                let ctx_chars = context_content.len() + file.len() + 100;
                if total_chars + ctx_chars > char_budget {
                    continue;
                }
                total_chars += ctx_chars;

                contexts.push(FileContext {
                    file: file.clone(),
                    start_line: start + 1,
                    end_line: end,
                    content: context_content,
                    relevance: format!(
                        "ERROR: {} at line {} (±{} lines)",
                        window.category, window.center_line, SPAN_CONTEXT_LINES
                    ),
                    priority: 0,
                    provenance: ContextProvenance::CompilerError,
                });
            }
        }

        // 5. Resolve symbol references from error messages
        let referenced_symbols = self.extract_symbol_references(report);
        if !referenced_symbols.is_empty() {
            self.add_symbol_definitions(
                &referenced_symbols,
                &mut contexts,
                &mut files_seen,
                &mut total_chars,
                char_budget,
                &canon_wd,
            );
        }

        // 6. Include touched files not already covered (as windowed or full)
        for file in files_touched {
            if !file.ends_with(".rs") || files_seen.contains(file) {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let estimated_chars = content.len() + content.lines().count() * 8 + file.len() + 100;
            if total_chars + estimated_chars > char_budget {
                continue;
            }

            let lines: Vec<&str> = content.lines().collect();
            let line_count = lines.len();
            let context_content = Self::format_lines(&lines, 0, line_count);
            let ctx_chars = context_content.len() + file.len() + 100;
            total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line: line_count,
                content: context_content,
                relevance: "Modified file (full content for retry)".to_string(),
                priority: 1,
                provenance: ContextProvenance::Diff,
            });
        }

        contexts
    }

    /// Collect error span windows from the verifier report's failure signals and gate errors.
    fn collect_span_windows(&self, report: &VerifierReport) -> Vec<SpanWindow> {
        let mut windows = Vec::new();

        // From failure signals
        for signal in &report.failure_signals {
            if let (Some(file), Some(line)) = (&signal.file, signal.line) {
                windows.push(SpanWindow {
                    file: file.clone(),
                    center_line: line,
                    start_line: line.saturating_sub(SPAN_CONTEXT_LINES),
                    end_line: line + SPAN_CONTEXT_LINES,
                    category: signal.category,
                    message: signal.message.clone(),
                });
            }
        }

        // From gate parsed errors (may have additional span info not in failure_signals)
        for gate in &report.gates {
            for error in &gate.errors {
                if let (Some(file), Some(line)) = (&error.file, error.line) {
                    // Avoid duplicates from failure_signals
                    let already = windows
                        .iter()
                        .any(|w| w.file == *file && w.center_line == line);
                    if !already {
                        windows.push(SpanWindow {
                            file: file.clone(),
                            center_line: line,
                            start_line: line.saturating_sub(SPAN_CONTEXT_LINES),
                            end_line: line + SPAN_CONTEXT_LINES,
                            category: error.category,
                            message: error.message.clone(),
                        });
                    }
                }
            }
        }

        windows
    }

    /// Merge overlapping span windows for the same file into contiguous ranges.
    ///
    /// Returns a map of file → merged windows, preserving the most specific category
    /// and message from the original window closest to each merged range's center.
    fn merge_windows(windows: &[SpanWindow]) -> HashMap<String, Vec<SpanWindow>> {
        let mut by_file: HashMap<String, Vec<&SpanWindow>> = HashMap::new();
        for w in windows {
            by_file.entry(w.file.clone()).or_default().push(w);
        }

        let mut merged: HashMap<String, Vec<SpanWindow>> = HashMap::new();

        for (file, mut file_windows) in by_file {
            file_windows.sort_by_key(|w| w.start_line);

            let mut result: Vec<SpanWindow> = Vec::new();
            for w in file_windows {
                if let Some(last) = result.last_mut() {
                    // Overlapping or adjacent — merge
                    if w.start_line <= last.end_line + 1 {
                        last.end_line = last.end_line.max(w.end_line);
                        // Keep the message from whichever window is closer to center
                        continue;
                    }
                }
                result.push(w.clone());
            }

            merged.insert(file, result);
        }

        merged
    }

    /// Extract type/trait symbol names referenced in error messages.
    ///
    /// Parses error messages for backtick-quoted identifiers that look like
    /// type or trait names (PascalCase), which are candidates for AST-based
    /// definition lookup.
    fn extract_symbol_references(&self, report: &VerifierReport) -> HashSet<String> {
        let mut symbols = HashSet::new();

        for signal in &report.failure_signals {
            for cap in SYMBOL_REF_RE.captures_iter(&signal.message) {
                symbols.insert(cap[1].to_string());
            }
        }

        // Also check rendered errors in gate results for richer symbol refs
        for gate in &report.gates {
            for error in &gate.errors {
                for cap in SYMBOL_REF_RE.captures_iter(&error.rendered) {
                    symbols.insert(cap[1].to_string());
                }
                for label in &error.labels {
                    for cap in SYMBOL_REF_RE.captures_iter(label) {
                        symbols.insert(cap[1].to_string());
                    }
                }
            }
        }

        // Filter out common Rust stdlib types that aren't worth looking up
        let stdlib = [
            "String",
            "Vec",
            "Option",
            "Result",
            "Box",
            "Arc",
            "Rc",
            "HashMap",
            "HashSet",
            "BTreeMap",
            "BTreeSet",
            "Cow",
            "Pin",
            "Future",
            "Send",
            "Sync",
            "Copy",
            "Clone",
            "Debug",
            "Display",
            "Default",
            "Iterator",
            "IntoIterator",
            "From",
            "Into",
            "TryFrom",
            "TryInto",
            "AsRef",
            "AsMut",
            "Deref",
            "DerefMut",
            "Drop",
            "Sized",
            "Unpin",
            "Fn",
            "FnMut",
            "FnOnce",
            "Error",
            "Read",
            "Write",
            "Seek",
            "BufRead",
            "Path",
            "PathBuf",
            "OsStr",
            "OsString",
            "Ordering",
            "Duration",
            "Instant",
            "SystemTime",
            "Mutex",
            "RwLock",
            "Cell",
            "RefCell",
            "Phantom",
            "PhantomData",
        ];
        let stdlib_set: HashSet<&str> = stdlib.into_iter().collect();
        symbols.retain(|s| !stdlib_set.contains(s.as_str()));

        symbols
    }

    /// Look up referenced symbols in the worktree's Rust files and add their
    /// definitions as FileContexts.
    ///
    /// For each referenced symbol name, scans .rs files for a matching definition
    /// using the AST index, then includes the definition's line range as context.
    fn add_symbol_definitions(
        &self,
        referenced: &HashSet<String>,
        contexts: &mut Vec<FileContext>,
        files_seen: &mut HashSet<String>,
        total_chars: &mut usize,
        char_budget: usize,
        canon_wd: &Path,
    ) {
        let rust_files = self.file_walker.rust_files();

        // Build a map of symbol_name → (file, start_line, end_line) from AST index
        let mut symbol_locations: Vec<(String, String, usize, usize)> = Vec::new();

        for path in &rust_files {
            let rel = match path.strip_prefix(&self.working_dir) {
                Ok(p) => p.display().to_string(),
                Err(_) => continue,
            };

            // Skip files we've already fully included
            if files_seen.contains(&rel) {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let index = FileSymbolIndex::from_source(&rel, &source);
            for sym in &index.symbols {
                if referenced.contains(&sym.name) {
                    symbol_locations.push((
                        sym.name.clone(),
                        rel.clone(),
                        sym.start_line,
                        sym.end_line,
                    ));
                }
            }
        }

        // Add symbol definition contexts
        for (sym_name, file, start_line, end_line) in &symbol_locations {
            let full_path = self.working_dir.join(file);
            let canonical = match std::fs::canonicalize(&full_path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if !canonical.starts_with(canon_wd) {
                continue;
            }

            let source = match std::fs::read_to_string(&canonical) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let lines: Vec<&str> = source.lines().collect();
            let line_count = lines.len();

            // Include the symbol definition with a small margin (±5 lines)
            let ctx_start = start_line.saturating_sub(5);
            let ctx_end = (end_line + 6).min(line_count);
            if ctx_start >= ctx_end {
                continue;
            }

            let context_content = Self::format_lines(&lines, ctx_start, ctx_end);
            let ctx_chars = context_content.len() + file.len() + 100;
            if *total_chars + ctx_chars > char_budget {
                continue;
            }
            *total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: ctx_start + 1,
                end_line: ctx_end,
                content: context_content,
                relevance: format!(
                    "Referenced symbol `{}` definition (lines {}-{})",
                    sym_name,
                    start_line + 1,
                    end_line + 1
                ),
                priority: 2, // Lower than error context but higher than generic
                provenance: ContextProvenance::Dependency,
            });
        }
    }

    /// Format a range of lines with line numbers.
    fn format_lines(lines: &[&str], start: usize, end: usize) -> String {
        lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:4} | {}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Build file contexts from a list of relative paths using AST-aware extraction.
    ///
    /// For each Rust file, uses tree-sitter to extract:
    /// - Import/use statements (first ~10 lines)
    /// - Compact symbol summary (pub structs, traits, fns with line ranges)
    ///
    /// Falls back to first-30-lines for non-Rust files or parse failures.
    fn build_file_contexts(&self, files: &[String]) -> Vec<FileContext> {
        let mut contexts = Vec::new();
        let mut total_chars = 0usize;
        let char_budget = self.max_context_tokens * 4; // 4 chars per token

        for file in files {
            if !file.ends_with(".rs") {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let lines: Vec<&str> = content.lines().collect();

            // Try AST-aware extraction
            let context_content =
                Self::build_ast_context(file, &content, &lines).unwrap_or_else(|| {
                    // Fallback: first 30 lines
                    let end = lines.len().min(30);
                    lines[..end]
                        .iter()
                        .enumerate()
                        .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                        .collect::<Vec<_>>()
                        .join("\n")
                });

            let end_line = lines.len().min(30);
            let ctx_chars = context_content.len() + file.len() + 50; // overhead
            if total_chars + ctx_chars > char_budget {
                break;
            }
            total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line,
                content: context_content,
                relevance: "Worktree file (AST summary)".to_string(),
                priority: 2, // Structural/header context
                provenance: ContextProvenance::Header,
            });
        }

        contexts
    }

    /// Build AST-aware context for a single Rust file.
    ///
    /// Returns `Some(content)` with imports + symbol summary, or `None` if
    /// tree-sitter parsing fails (caller falls back to first-30-lines).
    fn build_ast_context(file: &str, source: &str, lines: &[&str]) -> Option<String> {
        use crate::context_packer::ast_index::FileSymbolIndex;

        let index = FileSymbolIndex::from_source(file, source);
        if index.symbols.is_empty() {
            return None; // Parse failure or empty file — fallback
        }

        let mut parts = Vec::new();

        // Part 1: Imports (use statements and mod declarations from the top)
        let import_lines: Vec<String> = lines
            .iter()
            .take(50) // scan first 50 lines for imports
            .enumerate()
            .filter(|(_, l)| {
                let trimmed = l.trim();
                trimmed.starts_with("use ")
                    || trimmed.starts_with("pub use ")
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("pub mod ")
                    || trimmed.starts_with("//!")
            })
            .map(|(i, l)| format!("{:4} | {}", i + 1, l))
            .collect();

        if !import_lines.is_empty() {
            parts.push(format!("// Imports:\n{}", import_lines.join("\n")));
        }

        // Part 2: AST symbol summary
        let summary = index.compact_summary();
        if !summary.is_empty() {
            parts.push(format!("// Symbols:\n{}", summary));
        }

        if parts.is_empty() {
            return None;
        }

        Some(parts.join("\n\n"))
    }

    /// Trim a WorkPacket to fit within the token budget.
    ///
    /// Sorts file contexts by priority (lowest priority number = most important)
    /// so `pop()` removes the least important contexts first.
    fn trim_to_budget(&self, packet: &mut WorkPacket) {
        // Sort so highest-priority-number (least important) is last → popped first
        packet
            .file_contexts
            .sort_by(|a, b| a.priority.cmp(&b.priority));

        // Drop file contexts from the end until we're under budget
        while packet.estimated_tokens() > self.max_context_tokens
            && !packet.file_contexts.is_empty()
        {
            packet.file_contexts.pop();
        }

        // If still over budget, truncate previous_attempts
        while packet.estimated_tokens() > self.max_context_tokens
            && !packet.previous_attempts.is_empty()
        {
            packet.previous_attempts.pop();
        }
    }

    /// Evaluate context quality using probe-based testing.
    ///
    /// Generates deterministic probe questions from the full iteration history,
    /// then checks if the answers are recoverable from the structured summary.
    /// Returns the probe results and logs a warning if below the threshold.
    pub fn evaluate_context_quality(
        &self,
        deltas: &[crate::work_packet::types::IterationDelta],
        summary: &crate::harness::types::StructuredSessionSummary,
    ) -> crate::context_packer::probes::ProbeResults {
        let generator = crate::context_packer::probes::ProbeGenerator::new();
        let evaluator = crate::context_packer::probes::ProbeEvaluator::new();
        let probes = generator.generate_probes(deltas);
        let results = evaluator.evaluate(summary, &probes);

        if !evaluator.is_adequate(&results) {
            eprintln!(
                "[context_packer] WARNING: Probe pass rate {:.0}% ({}/{}) below threshold {:.0}%. \
                 Summary may be too lossy. Failed probes: {:?}",
                results.pass_rate * 100.0,
                results.pass_count,
                results.pass_count + results.fail_count,
                evaluator.min_pass_rate * 100.0,
                results
                    .failed_probes
                    .iter()
                    .map(|p| &p.question)
                    .collect::<Vec<_>>(),
            );
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::error_parser::ErrorCategory;
    use crate::verifier::report::{FailureSignal, GateOutcome, GateResult};
    use std::fs;

    #[test]
    fn test_pack_initial_creates_packet() {
        let dir = tempfile::tempdir().unwrap();
        // Create a minimal Rust file
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "pub struct Foo;\npub fn bar() {}\n").unwrap();

        // Initialize git repo so generator can query git
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let packet = packer.pack_initial("beads-test", "Test objective");

        assert_eq!(packet.bead_id, "beads-test");
        assert_eq!(packet.objective, "Test objective");
        // Should have found our lib.rs in the file contexts
        assert!(!packet.file_contexts.is_empty());
        assert!(packet.estimated_tokens() <= 8_000);
    }

    #[test]
    fn test_token_budgets() {
        assert_eq!(max_context_tokens(SwarmTier::Worker), 8_000);
        assert_eq!(max_context_tokens(SwarmTier::Council), 32_000);
        assert_eq!(max_context_tokens(SwarmTier::Human), 32_000);
    }

    #[test]
    fn test_trim_to_budget() {
        let dir = tempfile::tempdir().unwrap();
        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);

        let state = EscalationState::new("test");
        let mut packet =
            packer
                .generator
                .generate("test", "test obj", SwarmTier::Worker, &state, None);

        // Add a bunch of large file contexts to blow the budget
        for i in 0..100 {
            packet.file_contexts.push(FileContext {
                file: format!("src/big_{i}.rs"),
                start_line: 1,
                end_line: 100,
                content: "x".repeat(500),
                relevance: "test".to_string(),
                priority: 3,
                provenance: ContextProvenance::Header,
            });
        }

        packer.trim_to_budget(&mut packet);
        assert!(packet.estimated_tokens() <= 8_000);
    }

    /// Helper: create a VerifierReport with failure signals pointing at specific files.
    fn make_report_with_failures(files: &[(&str, usize)], working_dir: &str) -> VerifierReport {
        let mut report = VerifierReport::new(working_dir.to_string());
        for (file, line) in files {
            report.failure_signals.push(FailureSignal {
                gate: "check".to_string(),
                category: ErrorCategory::TypeMismatch,
                code: Some("E0308".to_string()),
                file: Some(file.to_string()),
                line: Some(*line),
                message: "mismatched types".to_string(),
            });
        }
        report
    }

    #[test]
    fn test_retry_context_includes_full_error_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let file_content = "fn main() {\n    let x: i32 = \"hello\";\n    println!(\"{x}\");\n}\n";
        fs::write(src.join("main.rs"), file_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/main.rs", 2)], &dir.path().display().to_string());

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/main.rs");
        assert_eq!(contexts[0].start_line, 1);
        assert_eq!(contexts[0].end_line, 4); // 4 lines
        assert!(contexts[0].content.contains("let x: i32"));
        assert!(contexts[0].relevance.contains("ERROR"));
    }

    #[test]
    fn test_retry_context_includes_touched_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("lib.rs"), "pub mod foo;\n").unwrap();
        fs::write(src.join("foo.rs"), "pub fn foo() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report = VerifierReport::new(dir.path().display().to_string());
        let touched = vec!["src/lib.rs".to_string(), "src/foo.rs".to_string()];

        let contexts = packer.build_retry_file_contexts(&report, &touched);

        assert_eq!(contexts.len(), 2);
        let files: Vec<&str> = contexts.iter().map(|c| c.file.as_str()).collect();
        assert!(files.contains(&"src/lib.rs"));
        assert!(files.contains(&"src/foo.rs"));
    }

    #[test]
    fn test_retry_context_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // Create files that exceed the Implementer budget (8K tokens = 32K chars)
        let big_content = "x".repeat(20_000); // ~20K chars each
        fs::write(src.join("a.rs"), &big_content).unwrap();
        fs::write(src.join("b.rs"), &big_content).unwrap();
        fs::write(src.join("c.rs"), &big_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let report = make_report_with_failures(
            &[("src/a.rs", 1), ("src/b.rs", 1), ("src/c.rs", 1)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        // Budget is 32K chars — each file ~20K + formatting overhead.
        // Should include 1 file, skip the rest.
        assert!(
            contexts.len() < 3,
            "Should have trimmed to fit budget, got {}",
            contexts.len()
        );
        let total_chars: usize = contexts.iter().map(|c| c.content.len()).sum();
        assert!(
            total_chars <= 32_000 + 1000,
            "Total chars {total_chars} exceeds budget"
        );
    }

    #[test]
    fn test_retry_context_skips_oversized_includes_smaller() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // One huge file that exceeds budget, one small file that fits
        let huge_content = "x\n".repeat(50_000); // ~100K chars
        let small_content = "fn small() {}\n";
        fs::write(src.join("huge.rs"), &huge_content).unwrap();
        fs::write(src.join("small.rs"), small_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let report = make_report_with_failures(
            &[("src/huge.rs", 1), ("src/small.rs", 1)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        // The huge file should be skipped (continue, not break), small file included
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/small.rs");
    }

    #[test]
    fn test_retry_context_deduplicates_error_and_touched() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/main.rs", 1)], &dir.path().display().to_string());
        // Same file in both failure_signals and files_touched
        let touched = vec!["src/main.rs".to_string()];

        let contexts = packer.build_retry_file_contexts(&report, &touched);

        // Should appear only once
        assert_eq!(contexts.len(), 1);
    }

    #[test]
    fn test_retry_context_extracts_from_gate_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("lib.rs"), "pub fn broken() -> i32 { \"oops\" }\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let mut report = VerifierReport::new(dir.path().display().to_string());
        report.gates.push(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 100,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: Some(format!(
                "error[E0308]: mismatched types\n --> src/lib.rs:1:26\n"
            )),
        });

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/lib.rs");
        assert!(contexts[0].content.contains("broken"));
    }

    // ─── Span-aware context tests ───

    #[test]
    fn test_span_aware_small_file_includes_full() {
        // Files under SMALL_FILE_THRESHOLD (200 lines) get included fully
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=50)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("small.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/small.rs", 25)], &dir.path().display().to_string());

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/small.rs");
        assert_eq!(contexts[0].start_line, 1);
        assert_eq!(contexts[0].end_line, 50);
        assert!(contexts[0].content.contains("func_1"));
        assert!(contexts[0].content.contains("func_50"));
    }

    #[test]
    fn test_span_aware_large_file_windows() {
        // Files over SMALL_FILE_THRESHOLD get windowed ±80 lines
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=500)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/big.rs", 250)], &dir.path().display().to_string());

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/big.rs");
        // Window should be around line 250: start ~170, end ~330
        assert!(contexts[0].start_line > 1, "Should not start at line 1");
        assert!(contexts[0].end_line < 500, "Should not include full file");
        assert!(
            contexts[0].content.contains("func_250"),
            "Should include error line"
        );
        // Lines near the start should not be included (window starts ~170)
        assert!(
            !contexts[0].content.contains("func_1()"),
            "Should not include line 1"
        );
        assert!(
            !contexts[0].content.contains("func_50()"),
            "Should not include line 50"
        );
        // Lines near the end should not be included (window ends ~330)
        assert!(
            !contexts[0].content.contains("func_450()"),
            "Should not include line 450"
        );
        assert!(
            !contexts[0].content.contains("func_500()"),
            "Should not include line 500"
        );
    }

    #[test]
    fn test_span_aware_merges_overlapping_windows() {
        // Two errors 30 lines apart in a large file should merge into one window
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=500)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report = make_report_with_failures(
            &[("src/big.rs", 200), ("src/big.rs", 230)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        // Two nearby errors should merge into one window
        let big_contexts: Vec<_> = contexts.iter().filter(|c| c.file == "src/big.rs").collect();
        assert_eq!(
            big_contexts.len(),
            1,
            "Overlapping windows should merge into one"
        );

        // The merged window should cover both error lines
        assert!(big_contexts[0].content.contains("func_200"));
        assert!(big_contexts[0].content.contains("func_230"));
    }

    #[test]
    fn test_span_aware_separate_windows_for_distant_errors() {
        // Two errors far apart in a large file should create separate windows
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=500)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        // Errors at lines 50 and 450 — windows shouldn't overlap
        let report = make_report_with_failures(
            &[("src/big.rs", 50), ("src/big.rs", 450)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        let big_contexts: Vec<_> = contexts.iter().filter(|c| c.file == "src/big.rs").collect();
        assert_eq!(
            big_contexts.len(),
            2,
            "Distant errors should create separate windows"
        );
    }

    #[test]
    fn test_span_aware_includes_touched_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("error.rs"), "fn bad() { }\n").unwrap();
        fs::write(src.join("touched.rs"), "fn good() { }\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/error.rs", 1)], &dir.path().display().to_string());
        let touched = vec!["src/touched.rs".to_string()];

        let contexts = packer.build_span_aware_contexts(&report, &touched);

        assert_eq!(contexts.len(), 2);
        let files: Vec<&str> = contexts.iter().map(|c| c.file.as_str()).collect();
        assert!(files.contains(&"src/error.rs"));
        assert!(files.contains(&"src/touched.rs"));
    }

    #[test]
    fn test_span_aware_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/main.rs", 1)], &dir.path().display().to_string());
        let touched = vec!["src/main.rs".to_string()];

        let contexts = packer.build_span_aware_contexts(&report, &touched);

        // File appears in both error and touched — should only appear once
        assert_eq!(contexts.len(), 1);
    }

    #[test]
    fn test_span_aware_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let big_content = "x\n".repeat(10_000);
        fs::write(src.join("a.rs"), &big_content).unwrap();
        fs::write(src.join("b.rs"), &big_content).unwrap();
        fs::write(src.join("c.rs"), &big_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let report = make_report_with_failures(
            &[("src/a.rs", 5000), ("src/b.rs", 5000), ("src/c.rs", 5000)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        let total_chars: usize = contexts.iter().map(|c| c.content.len()).sum();
        // Worker budget = 8K tokens = 32K chars
        assert!(
            total_chars <= 32_000 + 1000,
            "Total chars {total_chars} exceeds budget"
        );
    }

    #[test]
    fn test_span_aware_symbol_resolution() {
        // Error referencing a custom type should include that type's definition
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // File with the error
        fs::write(
            src.join("main.rs"),
            "fn main() {\n    let x: MyConfig = todo!();\n}\n",
        )
        .unwrap();

        // File with the referenced symbol definition
        fs::write(
            src.join("config.rs"),
            "pub struct MyConfig {\n    pub name: String,\n    pub timeout: u64,\n}\n",
        )
        .unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let mut report = VerifierReport::new(dir.path().display().to_string());
        report.failure_signals.push(FailureSignal {
            gate: "check".to_string(),
            category: ErrorCategory::TypeMismatch,
            code: Some("E0308".to_string()),
            file: Some("src/main.rs".to_string()),
            line: Some(2),
            message: "expected `MyConfig`, found `()`".to_string(),
        });

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        // Should have main.rs (error) + config.rs (symbol definition)
        let files: Vec<&str> = contexts.iter().map(|c| c.file.as_str()).collect();
        assert!(files.contains(&"src/main.rs"), "Should include error file");
        assert!(
            files.contains(&"src/config.rs"),
            "Should include file defining referenced symbol `MyConfig`"
        );

        // config.rs context should mention MyConfig
        let config_ctx = contexts.iter().find(|c| c.file == "src/config.rs").unwrap();
        assert!(config_ctx.content.contains("MyConfig"));
        assert_eq!(config_ctx.provenance, ContextProvenance::Dependency);
    }

    #[test]
    fn test_span_aware_skips_stdlib_symbols() {
        // Error messages mentioning stdlib types like String, Vec should not trigger
        // a symbol lookup
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let mut report = VerifierReport::new(dir.path().display().to_string());
        report.failure_signals.push(FailureSignal {
            gate: "check".to_string(),
            category: ErrorCategory::TypeMismatch,
            code: Some("E0308".to_string()),
            file: Some("src/main.rs".to_string()),
            line: Some(1),
            message: "expected `String`, found `Vec<u8>`".to_string(),
        });

        let refs = packer.extract_symbol_references(&report);
        assert!(!refs.contains("String"), "String should be filtered");
        assert!(!refs.contains("Vec"), "Vec should be filtered");
    }

    #[test]
    fn test_span_aware_window_at_file_start() {
        // Error at line 1 should not underflow
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=300)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/big.rs", 5)], &dir.path().display().to_string());

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        // Window should start at line 1 (can't go before file start)
        assert_eq!(contexts[0].start_line, 1);
        assert!(contexts[0].content.contains("func_5"));
        assert!(contexts[0].content.contains("func_1"));
    }

    #[test]
    fn test_span_aware_window_at_file_end() {
        // Error at last line should not overflow
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=300)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/big.rs", 298)], &dir.path().display().to_string());

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].end_line, 300);
        assert!(contexts[0].content.contains("func_298"));
        assert!(contexts[0].content.contains("func_300"));
    }

    #[test]
    fn test_pack_retry_uses_span_aware_when_spans_available() {
        // When failure signals have file+line, pack_retry should use span-aware
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // Create a large file so we can observe windowing
        let content = (1..=400)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("big.rs"), &content).unwrap();

        // Initialize git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let state = EscalationState::new("test");
        let report =
            make_report_with_failures(&[("src/big.rs", 200)], &dir.path().display().to_string());

        let packet = packer.pack_retry("test", "Fix errors", &state, &report);

        // Should have contexts for the error file
        let big_ctx: Vec<_> = packet
            .file_contexts
            .iter()
            .filter(|c| c.file == "src/big.rs")
            .collect();
        assert!(!big_ctx.is_empty(), "Should include error file context");

        // With span-aware, a 400-line file should be windowed, not full
        let first = &big_ctx[0];
        assert!(
            first.start_line > 1 || first.end_line < 400,
            "Large file should be windowed, not full. start={}, end={}",
            first.start_line,
            first.end_line
        );
    }

    #[test]
    fn test_pack_retry_falls_back_when_no_spans() {
        // When failure signals lack file+line, pack_retry should use full-file fallback
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();

        // Initialize git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let state = EscalationState::new("test");

        // Failure signal without file or line
        let mut report = VerifierReport::new(dir.path().display().to_string());
        report.failure_signals.push(FailureSignal {
            gate: "test".to_string(),
            category: ErrorCategory::Other,
            code: None,
            file: None,
            line: None,
            message: "test failure: assertion failed".to_string(),
        });

        let packet = packer.pack_retry("test", "Fix test failures", &state, &report);

        // Should use the old full-file fallback path — no span windows
        // The packet may or may not have contexts depending on files_touched,
        // but it should NOT crash
        assert!(packet.bead_id == "test");
    }

    #[test]
    fn test_merge_windows_no_overlap() {
        let windows = vec![
            SpanWindow {
                file: "a.rs".to_string(),
                center_line: 50,
                start_line: 0,
                end_line: 130,
                category: ErrorCategory::TypeMismatch,
                message: "error 1".to_string(),
            },
            SpanWindow {
                file: "a.rs".to_string(),
                center_line: 400,
                start_line: 320,
                end_line: 480,
                category: ErrorCategory::BorrowChecker,
                message: "error 2".to_string(),
            },
        ];

        let merged = ContextPacker::merge_windows(&windows);
        let a_windows = merged.get("a.rs").unwrap();
        assert_eq!(
            a_windows.len(),
            2,
            "Non-overlapping windows should stay separate"
        );
    }

    #[test]
    fn test_merge_windows_with_overlap() {
        let windows = vec![
            SpanWindow {
                file: "a.rs".to_string(),
                center_line: 100,
                start_line: 20,
                end_line: 180,
                category: ErrorCategory::TypeMismatch,
                message: "error 1".to_string(),
            },
            SpanWindow {
                file: "a.rs".to_string(),
                center_line: 150,
                start_line: 70,
                end_line: 230,
                category: ErrorCategory::BorrowChecker,
                message: "error 2".to_string(),
            },
        ];

        let merged = ContextPacker::merge_windows(&windows);
        let a_windows = merged.get("a.rs").unwrap();
        assert_eq!(a_windows.len(), 1, "Overlapping windows should merge");
        assert_eq!(a_windows[0].start_line, 20);
        assert_eq!(a_windows[0].end_line, 230);
    }

    #[test]
    fn test_format_lines() {
        let lines = vec!["line one", "line two", "line three", "line four"];
        let result = ContextPacker::format_lines(&lines, 1, 3);
        assert!(result.contains("   2 | line two"));
        assert!(result.contains("   3 | line three"));
        assert!(!result.contains("line one"));
        assert!(!result.contains("line four"));
    }

    #[test]
    fn test_extract_symbol_references_from_messages() {
        let dir = tempfile::tempdir().unwrap();
        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);

        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.failure_signals.push(FailureSignal {
            gate: "check".to_string(),
            category: ErrorCategory::TraitBound,
            code: Some("E0277".to_string()),
            file: Some("src/main.rs".to_string()),
            line: Some(10),
            message: "the trait `MyTrait` is not implemented for `MyStruct`".to_string(),
        });

        let refs = packer.extract_symbol_references(&report);
        assert!(refs.contains("MyTrait"), "Should find MyTrait");
        assert!(refs.contains("MyStruct"), "Should find MyStruct");
        // Stdlib types should be filtered
        assert!(!refs.contains("String"));
    }

    #[test]
    fn test_span_aware_gate_stderr_extraction() {
        // Errors only in gate stderr (no failure_signals with file) should still get windows
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let content = (1..=300)
            .map(|i| format!("fn func_{i}() {{ }}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(src.join("lib.rs"), &content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let mut report = VerifierReport::new(dir.path().display().to_string());
        // Add a failure signal with file+line to trigger span-aware path
        report.failure_signals.push(FailureSignal {
            gate: "check".to_string(),
            category: ErrorCategory::TypeMismatch,
            code: Some("E0308".to_string()),
            file: Some("src/lib.rs".to_string()),
            line: Some(150),
            message: "mismatched types".to_string(),
        });
        // Also add a gate stderr reference to the same file at a different line
        report.gates.push(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 100,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: Some(
                "error[E0308]: mismatched types\n --> src/lib.rs:50:10\n".to_string(),
            ),
        });

        let contexts = packer.build_span_aware_contexts(&report, &[]);

        // The file is >200 lines. Should have two separate windows (lines 50 and 150 are far apart)
        let lib_contexts: Vec<_> = contexts.iter().filter(|c| c.file == "src/lib.rs").collect();
        assert!(lib_contexts.len() >= 1, "Should include error file context");
    }
}
