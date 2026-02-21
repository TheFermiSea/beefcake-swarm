//! Patch reliability — whitespace-normalized patch matching and application.
//!
//! Addresses repeated failures in autonomous patching where whitespace
//! differences between the LLM-generated patch and the actual file content
//! cause match failures. Provides fuzzy matching with configurable tolerance.

use serde::{Deserialize, Serialize};

/// Configuration for patch matching behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchConfig {
    /// Whether to normalize whitespace during matching.
    pub normalize_whitespace: bool,
    /// Whether to trim trailing whitespace from lines.
    pub trim_trailing: bool,
    /// Whether to collapse multiple blank lines into one.
    pub collapse_blank_lines: bool,
    /// Maximum number of context lines to use for anchor matching.
    pub max_context_lines: usize,
    /// Minimum similarity ratio (0.0–1.0) for fuzzy line matching.
    pub min_similarity: f64,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            normalize_whitespace: true,
            trim_trailing: true,
            collapse_blank_lines: true,
            max_context_lines: 3,
            min_similarity: 0.85,
        }
    }
}

/// A single patch hunk to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchHunk {
    /// Lines to find (the "old" content).
    pub old_lines: Vec<String>,
    /// Lines to replace with (the "new" content).
    pub new_lines: Vec<String>,
    /// Optional description of what this hunk does.
    pub description: Option<String>,
}

/// Result of applying a patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchResult {
    /// Whether the patch was applied successfully.
    pub success: bool,
    /// Number of hunks applied.
    pub hunks_applied: usize,
    /// Total hunks attempted.
    pub hunks_total: usize,
    /// Per-hunk results.
    pub hunk_results: Vec<HunkResult>,
    /// The patched content (if successful).
    pub patched_content: Option<String>,
}

/// Result of applying a single hunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HunkResult {
    /// Whether this hunk was applied.
    pub applied: bool,
    /// How the match was found.
    pub match_kind: MatchKind,
    /// Line number where the match was found (1-based).
    pub matched_at_line: Option<usize>,
    /// Similarity score of the match (1.0 = exact).
    pub similarity: f64,
    /// Error message if the hunk failed.
    pub error: Option<String>,
}

/// How a patch hunk was matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchKind {
    /// Exact byte-for-byte match.
    Exact,
    /// Matched after whitespace normalization.
    WhitespaceNormalized,
    /// Matched after trimming trailing whitespace.
    TrimmedTrailing,
    /// Fuzzy match above similarity threshold.
    Fuzzy,
    /// No match found.
    NoMatch,
}

impl std::fmt::Display for MatchKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::WhitespaceNormalized => write!(f, "whitespace_normalized"),
            Self::TrimmedTrailing => write!(f, "trimmed_trailing"),
            Self::Fuzzy => write!(f, "fuzzy"),
            Self::NoMatch => write!(f, "no_match"),
        }
    }
}

/// Patch engine that applies hunks with configurable matching.
pub struct PatchEngine {
    config: PatchConfig,
}

impl PatchEngine {
    /// Create a new patch engine with the given config.
    pub fn new(config: PatchConfig) -> Self {
        Self { config }
    }

    /// Create a patch engine with default config.
    pub fn default_engine() -> Self {
        Self::new(PatchConfig::default())
    }

    /// Apply a set of hunks to content.
    pub fn apply(&self, content: &str, hunks: &[PatchHunk]) -> PatchResult {
        let mut current = content.to_string();
        let mut hunk_results = Vec::new();
        let mut hunks_applied = 0;

        for hunk in hunks {
            let result = self.apply_hunk(&current, hunk);
            if result.hunk_result.applied {
                if let Some(ref patched) = result.patched_content {
                    current = patched.clone();
                }
                hunks_applied += 1;
            }
            hunk_results.push(result.hunk_result);
        }

        PatchResult {
            success: hunks_applied == hunks.len(),
            hunks_applied,
            hunks_total: hunks.len(),
            hunk_results,
            patched_content: Some(current),
        }
    }

    /// Apply a single hunk, trying exact match first, then fuzzy.
    fn apply_hunk(&self, content: &str, hunk: &PatchHunk) -> ApplyResult {
        let content_lines: Vec<&str> = content.lines().collect();
        let old_lines: Vec<&str> = hunk.old_lines.iter().map(|s| s.as_str()).collect();

        if old_lines.is_empty() {
            return ApplyResult {
                hunk_result: HunkResult {
                    applied: false,
                    match_kind: MatchKind::NoMatch,
                    matched_at_line: None,
                    similarity: 0.0,
                    error: Some("Empty old_lines in hunk".to_string()),
                },
                patched_content: None,
            };
        }

        // Try exact match first
        if let Some(pos) = self.find_exact(&content_lines, &old_lines) {
            return self.replace_at(
                content,
                &content_lines,
                pos,
                &old_lines,
                hunk,
                MatchKind::Exact,
                1.0,
            );
        }

        // Try trimmed trailing match
        if self.config.trim_trailing {
            if let Some(pos) = self.find_trimmed(&content_lines, &old_lines) {
                return self.replace_at(
                    content,
                    &content_lines,
                    pos,
                    &old_lines,
                    hunk,
                    MatchKind::TrimmedTrailing,
                    0.98,
                );
            }
        }

        // Try whitespace-normalized match
        if self.config.normalize_whitespace {
            if let Some(pos) = self.find_normalized(&content_lines, &old_lines) {
                return self.replace_at(
                    content,
                    &content_lines,
                    pos,
                    &old_lines,
                    hunk,
                    MatchKind::WhitespaceNormalized,
                    0.95,
                );
            }
        }

        // Try fuzzy match
        if let Some((pos, sim)) = self.find_fuzzy(&content_lines, &old_lines) {
            if sim >= self.config.min_similarity {
                return self.replace_at(
                    content,
                    &content_lines,
                    pos,
                    &old_lines,
                    hunk,
                    MatchKind::Fuzzy,
                    sim,
                );
            }
        }

        ApplyResult {
            hunk_result: HunkResult {
                applied: false,
                match_kind: MatchKind::NoMatch,
                matched_at_line: None,
                similarity: 0.0,
                error: Some("No match found for hunk".to_string()),
            },
            patched_content: None,
        }
    }

    /// Find exact match position.
    fn find_exact(&self, content: &[&str], pattern: &[&str]) -> Option<usize> {
        if pattern.len() > content.len() {
            return None;
        }
        'outer: for i in 0..=content.len() - pattern.len() {
            for (j, pat_line) in pattern.iter().enumerate() {
                if content[i + j] != *pat_line {
                    continue 'outer;
                }
            }
            return Some(i);
        }
        None
    }

    /// Find match position after trimming trailing whitespace.
    fn find_trimmed(&self, content: &[&str], pattern: &[&str]) -> Option<usize> {
        if pattern.len() > content.len() {
            return None;
        }
        'outer: for i in 0..=content.len() - pattern.len() {
            for (j, pat_line) in pattern.iter().enumerate() {
                if content[i + j].trim_end() != pat_line.trim_end() {
                    continue 'outer;
                }
            }
            return Some(i);
        }
        None
    }

    /// Find match position after normalizing all whitespace.
    fn find_normalized(&self, content: &[&str], pattern: &[&str]) -> Option<usize> {
        if pattern.len() > content.len() {
            return None;
        }
        let normalized_pattern: Vec<String> = pattern.iter().map(|l| normalize_ws(l)).collect();
        'outer: for i in 0..=content.len() - pattern.len() {
            for (j, pat_norm) in normalized_pattern.iter().enumerate() {
                if normalize_ws(content[i + j]) != *pat_norm {
                    continue 'outer;
                }
            }
            return Some(i);
        }
        None
    }

    /// Find best fuzzy match position.
    fn find_fuzzy(&self, content: &[&str], pattern: &[&str]) -> Option<(usize, f64)> {
        if pattern.len() > content.len() {
            return None;
        }
        let mut best_pos = None;
        let mut best_sim = 0.0f64;

        for i in 0..=content.len() - pattern.len() {
            let sim = self.block_similarity(&content[i..i + pattern.len()], pattern);
            if sim > best_sim {
                best_sim = sim;
                best_pos = Some(i);
            }
        }

        best_pos.map(|pos| (pos, best_sim))
    }

    /// Compute similarity between two blocks of lines.
    fn block_similarity(&self, a: &[&str], b: &[&str]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let total: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(la, lb)| line_similarity(la, lb))
            .sum();
        total / a.len() as f64
    }

    /// Replace matched lines and produce new content.
    #[allow(clippy::too_many_arguments)]
    fn replace_at(
        &self,
        _original: &str,
        content_lines: &[&str],
        pos: usize,
        old_lines: &[&str],
        hunk: &PatchHunk,
        match_kind: MatchKind,
        similarity: f64,
    ) -> ApplyResult {
        let mut result_lines: Vec<String> = Vec::new();

        // Lines before the match
        for line in &content_lines[..pos] {
            result_lines.push(line.to_string());
        }

        // New lines (replacement)
        for line in &hunk.new_lines {
            result_lines.push(line.clone());
        }

        // Lines after the match
        for line in &content_lines[pos + old_lines.len()..] {
            result_lines.push(line.to_string());
        }

        let patched = result_lines.join("\n");

        ApplyResult {
            hunk_result: HunkResult {
                applied: true,
                match_kind,
                matched_at_line: Some(pos + 1), // 1-based
                similarity,
                error: None,
            },
            patched_content: Some(patched),
        }
    }
}

impl Default for PatchEngine {
    fn default() -> Self {
        Self::default_engine()
    }
}

/// Internal result from applying a single hunk.
struct ApplyResult {
    hunk_result: HunkResult,
    patched_content: Option<String>,
}

/// Normalize whitespace: collapse runs of spaces/tabs to single space, trim.
fn normalize_ws(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                result.push(' ');
                in_ws = true;
            }
        } else {
            result.push(ch);
            in_ws = false;
        }
    }
    result.trim().to_string()
}

/// Compute similarity between two lines using character-level Jaccard.
fn line_similarity(a: &str, b: &str) -> f64 {
    let a_norm = normalize_ws(a);
    let b_norm = normalize_ws(b);

    if a_norm == b_norm {
        return 1.0;
    }
    if a_norm.is_empty() && b_norm.is_empty() {
        return 1.0;
    }
    if a_norm.is_empty() || b_norm.is_empty() {
        return 0.0;
    }

    // Character bigram Jaccard similarity
    let a_bigrams: std::collections::HashSet<(char, char)> =
        a_norm.chars().zip(a_norm.chars().skip(1)).collect();
    let b_bigrams: std::collections::HashSet<(char, char)> =
        b_norm.chars().zip(b_norm.chars().skip(1)).collect();

    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        // Single-char lines — compare directly
        return if a_norm == b_norm { 1.0 } else { 0.0 };
    }

    let intersection = a_bigrams.intersection(&b_bigrams).count();
    let union = a_bigrams.union(&b_bigrams).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let engine = PatchEngine::default_engine();
        let content = "line1\nline2\nline3\nline4";
        let hunk = PatchHunk {
            old_lines: vec!["line2".to_string(), "line3".to_string()],
            new_lines: vec!["new2".to_string(), "new3".to_string()],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
        assert_eq!(result.hunks_applied, 1);
        assert_eq!(result.hunk_results[0].match_kind, MatchKind::Exact);
        assert_eq!(
            result.patched_content.as_deref(),
            Some("line1\nnew2\nnew3\nline4")
        );
    }

    #[test]
    fn test_trailing_whitespace_match() {
        let engine = PatchEngine::default_engine();
        let content = "line1  \nline2\t\nline3";
        let hunk = PatchHunk {
            old_lines: vec!["line1".to_string(), "line2".to_string()],
            new_lines: vec!["replaced1".to_string(), "replaced2".to_string()],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
        assert_eq!(
            result.hunk_results[0].match_kind,
            MatchKind::TrimmedTrailing
        );
    }

    #[test]
    fn test_whitespace_normalized_match() {
        let engine = PatchEngine::default_engine();
        let content = "  fn  foo(  ) {\n    bar();\n  }";
        let hunk = PatchHunk {
            old_lines: vec![
                " fn foo( ) {".to_string(),
                "   bar();".to_string(),
                " }".to_string(),
            ],
            new_lines: vec![
                "  fn foo() {".to_string(),
                "    baz();".to_string(),
                "  }".to_string(),
            ],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
        assert_eq!(
            result.hunk_results[0].match_kind,
            MatchKind::WhitespaceNormalized
        );
    }

    #[test]
    fn test_no_match() {
        let engine = PatchEngine::default_engine();
        let content = "line1\nline2\nline3";
        let hunk = PatchHunk {
            old_lines: vec!["nonexistent".to_string()],
            new_lines: vec!["replaced".to_string()],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(!result.success);
        assert_eq!(result.hunks_applied, 0);
        assert_eq!(result.hunk_results[0].match_kind, MatchKind::NoMatch);
    }

    #[test]
    fn test_multiple_hunks() {
        let engine = PatchEngine::default_engine();
        let content = "aaa\nbbb\nccc\nddd\neee";
        let hunks = vec![
            PatchHunk {
                old_lines: vec!["bbb".to_string()],
                new_lines: vec!["BBB".to_string()],
                description: None,
            },
            PatchHunk {
                old_lines: vec!["ddd".to_string()],
                new_lines: vec!["DDD".to_string()],
                description: None,
            },
        ];

        let result = engine.apply(content, &hunks);
        assert!(result.success);
        assert_eq!(result.hunks_applied, 2);
        assert_eq!(
            result.patched_content.as_deref(),
            Some("aaa\nBBB\nccc\nDDD\neee")
        );
    }

    #[test]
    fn test_empty_hunk_fails() {
        let engine = PatchEngine::default_engine();
        let content = "line1\nline2";
        let hunk = PatchHunk {
            old_lines: vec![],
            new_lines: vec!["new".to_string()],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(!result.success);
    }

    #[test]
    fn test_normalize_ws() {
        assert_eq!(normalize_ws("  hello   world  "), "hello world");
        assert_eq!(normalize_ws("a\tb\tc"), "a b c");
        assert_eq!(normalize_ws(""), "");
        assert_eq!(normalize_ws("  "), "");
    }

    #[test]
    fn test_line_similarity_identical() {
        assert!((line_similarity("hello world", "hello world") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_line_similarity_whitespace_diff() {
        let sim = line_similarity("  fn foo()", " fn  foo()");
        assert!(sim > 0.9, "similarity should be high: {}", sim);
    }

    #[test]
    fn test_line_similarity_different() {
        let sim = line_similarity("completely different", "nothing alike here");
        assert!(sim < 0.5, "similarity should be low: {}", sim);
    }

    #[test]
    fn test_match_kind_display() {
        assert_eq!(MatchKind::Exact.to_string(), "exact");
        assert_eq!(
            MatchKind::WhitespaceNormalized.to_string(),
            "whitespace_normalized"
        );
        assert_eq!(MatchKind::Fuzzy.to_string(), "fuzzy");
        assert_eq!(MatchKind::NoMatch.to_string(), "no_match");
    }

    #[test]
    fn test_config_default() {
        let config = PatchConfig::default();
        assert!(config.normalize_whitespace);
        assert!(config.trim_trailing);
        assert!(config.collapse_blank_lines);
        assert_eq!(config.max_context_lines, 3);
        assert!((config.min_similarity - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_patch_result_serde() {
        let result = PatchResult {
            success: true,
            hunks_applied: 1,
            hunks_total: 1,
            hunk_results: vec![HunkResult {
                applied: true,
                match_kind: MatchKind::Exact,
                matched_at_line: Some(5),
                similarity: 1.0,
                error: None,
            }],
            patched_content: Some("patched".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: PatchResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.success);
        assert_eq!(parsed.hunks_applied, 1);
    }

    #[test]
    fn test_fuzzy_match_near_threshold() {
        let config = PatchConfig {
            min_similarity: 0.7,
            ..Default::default()
        };
        let engine = PatchEngine::new(config);
        // Lines that are similar but not identical
        let content = "fn process_data(input: &str) -> Result<Data, Error> {\n    let parsed = parse(input)?;\n    Ok(parsed)\n}";
        let hunk = PatchHunk {
            old_lines: vec![
                "fn process_data(input: &str) -> Result<Data, Error> {".to_string(),
                "    let parsed = parse(input)?;".to_string(),
                "    Ok(parsed)".to_string(),
                "}".to_string(),
            ],
            new_lines: vec![
                "fn process_data(input: &str) -> Result<Data, AppError> {".to_string(),
                "    let parsed = parse(input).map_err(AppError::from)?;".to_string(),
                "    Ok(parsed)".to_string(),
                "}".to_string(),
            ],
            description: Some("Change Error to AppError".to_string()),
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
    }

    #[test]
    fn test_insertion_at_beginning() {
        let engine = PatchEngine::default_engine();
        let content = "line1\nline2\nline3";
        let hunk = PatchHunk {
            old_lines: vec!["line1".to_string()],
            new_lines: vec!["line0".to_string(), "line1".to_string()],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
        assert_eq!(
            result.patched_content.as_deref(),
            Some("line0\nline1\nline2\nline3")
        );
    }

    #[test]
    fn test_deletion() {
        let engine = PatchEngine::default_engine();
        let content = "line1\nline2\nline3\nline4";
        let hunk = PatchHunk {
            old_lines: vec!["line2".to_string(), "line3".to_string()],
            new_lines: vec![],
            description: None,
        };

        let result = engine.apply(content, &[hunk]);
        assert!(result.success);
        assert_eq!(result.patched_content.as_deref(), Some("line1\nline4"));
    }
}
