//! ast-grep (sg) tool wrapper — bounded structural code search.
//!
//! Wraps the `sg` CLI with structured input/output, timeout enforcement,
//! and result truncation for safe use by the reviewer agent.

use serde::{Deserialize, Serialize};

/// Configuration for the ast-grep runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepConfig {
    /// Path to the sg binary (default: "sg").
    pub binary: String,
    /// Maximum execution time in milliseconds.
    pub timeout_ms: u64,
    /// Maximum number of matches to return.
    pub max_matches: usize,
    /// Maximum output bytes before truncation.
    pub max_output_bytes: usize,
    /// Working directory for sg execution.
    pub working_dir: Option<String>,
}

impl Default for AstGrepConfig {
    fn default() -> Self {
        Self {
            binary: "sg".to_string(),
            timeout_ms: 10_000,
            max_matches: 100,
            max_output_bytes: 65_536,
            working_dir: None,
        }
    }
}

/// A query for ast-grep search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepQuery {
    /// The pattern to search for (sg pattern syntax).
    pub pattern: String,
    /// Language to search (rust, typescript, python, etc.).
    pub language: String,
    /// File paths to search (empty = all).
    pub paths: Vec<String>,
    /// Optional rule ID for rule-based searches.
    pub rule_id: Option<String>,
    /// Whether to use --json output mode.
    pub json_output: bool,
}

impl AstGrepQuery {
    /// Create a new pattern-based query.
    pub fn pattern(pattern: &str, language: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            language: language.to_string(),
            paths: Vec::new(),
            rule_id: None,
            json_output: true,
        }
    }

    /// Create a rule-based query.
    pub fn rule(rule_id: &str, language: &str) -> Self {
        Self {
            pattern: String::new(),
            language: language.to_string(),
            paths: Vec::new(),
            rule_id: Some(rule_id.to_string()),
            json_output: true,
        }
    }

    /// Restrict to specific paths.
    pub fn in_paths(mut self, paths: Vec<String>) -> Self {
        self.paths = paths;
        self
    }

    /// Build the command-line arguments for sg.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        if let Some(ref rule_id) = self.rule_id {
            args.push("scan".to_string());
            args.push("--rule".to_string());
            args.push(rule_id.clone());
        } else {
            args.push("run".to_string());
            args.push("--pattern".to_string());
            args.push(self.pattern.clone());
        }

        args.push("--lang".to_string());
        args.push(self.language.clone());

        if self.json_output {
            args.push("--json".to_string());
        }

        for path in &self.paths {
            args.push(path.clone());
        }

        args
    }
}

/// A single match from ast-grep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepMatch {
    /// File where the match was found.
    pub file: String,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (1-indexed).
    pub column: u32,
    /// End line.
    pub end_line: u32,
    /// End column.
    pub end_column: u32,
    /// The matched text.
    pub text: String,
    /// Rule ID if from a rule scan.
    pub rule_id: Option<String>,
    /// Severity from rule (if applicable).
    pub severity: Option<String>,
    /// Message from rule (if applicable).
    pub message: Option<String>,
}

impl std::fmt::Display for AstGrepMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.file, self.line, self.column, self.text
        )
    }
}

/// Result from an ast-grep execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepResult {
    /// Matches found.
    pub matches: Vec<AstGrepMatch>,
    /// Total matches before truncation.
    pub total_matches: usize,
    /// Whether results were truncated.
    pub truncated: bool,
    /// Execution time in milliseconds.
    pub execution_ms: u64,
    /// Whether the query timed out.
    pub timed_out: bool,
    /// Error message (if any).
    pub error: Option<String>,
}

impl AstGrepResult {
    /// Create a successful result.
    pub fn ok(matches: Vec<AstGrepMatch>, execution_ms: u64) -> Self {
        let total = matches.len();
        Self {
            matches,
            total_matches: total,
            truncated: false,
            execution_ms,
            timed_out: false,
            error: None,
        }
    }

    /// Create a timeout result.
    pub fn timeout(timeout_ms: u64) -> Self {
        Self {
            matches: Vec::new(),
            total_matches: 0,
            truncated: false,
            execution_ms: timeout_ms,
            timed_out: true,
            error: Some(format!("timed out after {}ms", timeout_ms)),
        }
    }

    /// Create an error result.
    pub fn err(error: &str, execution_ms: u64) -> Self {
        Self {
            matches: Vec::new(),
            total_matches: 0,
            truncated: false,
            execution_ms,
            timed_out: false,
            error: Some(error.to_string()),
        }
    }

    /// Whether the search was successful.
    pub fn is_success(&self) -> bool {
        self.error.is_none() && !self.timed_out
    }

    /// Truncate to max matches.
    pub fn truncate_to(&mut self, max: usize) {
        if self.matches.len() > max {
            self.total_matches = self.matches.len();
            self.matches.truncate(max);
            self.truncated = true;
        }
    }

    /// Compact summary line.
    pub fn summary_line(&self) -> String {
        if let Some(ref err) = self.error {
            format!("[ERROR] {} ({}ms)", err, self.execution_ms)
        } else if self.truncated {
            format!(
                "[OK] {} matches (truncated from {}, {}ms)",
                self.matches.len(),
                self.total_matches,
                self.execution_ms
            )
        } else {
            format!(
                "[OK] {} matches ({}ms)",
                self.matches.len(),
                self.execution_ms
            )
        }
    }
}

/// Runner that manages ast-grep execution.
pub struct AstGrepRunner {
    config: AstGrepConfig,
}

impl AstGrepRunner {
    /// Create a new runner with default config.
    pub fn new() -> Self {
        Self {
            config: AstGrepConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: AstGrepConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &AstGrepConfig {
        &self.config
    }

    /// Validate a query before execution.
    pub fn validate_query(&self, query: &AstGrepQuery) -> Result<(), String> {
        if query.rule_id.is_none() && query.pattern.is_empty() {
            return Err("query must have either a pattern or rule_id".to_string());
        }
        if query.language.is_empty() {
            return Err("language is required".to_string());
        }
        Ok(())
    }

    /// Build a bounded result from raw matches, applying config limits.
    pub fn apply_bounds(&self, mut result: AstGrepResult) -> AstGrepResult {
        result.truncate_to(self.config.max_matches);
        result
    }
}

impl Default for AstGrepRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_query_args() {
        let query = AstGrepQuery::pattern("$EXPR.unwrap()", "rust");
        let args = query.to_args();
        assert!(args.contains(&"run".to_string()));
        assert!(args.contains(&"--pattern".to_string()));
        assert!(args.contains(&"$EXPR.unwrap()".to_string()));
        assert!(args.contains(&"--lang".to_string()));
        assert!(args.contains(&"rust".to_string()));
        assert!(args.contains(&"--json".to_string()));
    }

    #[test]
    fn test_rule_query_args() {
        let query = AstGrepQuery::rule("no-unwrap", "rust");
        let args = query.to_args();
        assert!(args.contains(&"scan".to_string()));
        assert!(args.contains(&"--rule".to_string()));
        assert!(args.contains(&"no-unwrap".to_string()));
    }

    #[test]
    fn test_query_with_paths() {
        let query = AstGrepQuery::pattern("$X", "rust")
            .in_paths(vec!["src/".to_string(), "tests/".to_string()]);
        let args = query.to_args();
        assert!(args.contains(&"src/".to_string()));
        assert!(args.contains(&"tests/".to_string()));
    }

    #[test]
    fn test_result_ok() {
        let matches = vec![AstGrepMatch {
            file: "src/lib.rs".to_string(),
            line: 10,
            column: 5,
            end_line: 10,
            end_column: 20,
            text: "x.unwrap()".to_string(),
            rule_id: None,
            severity: None,
            message: None,
        }];
        let result = AstGrepResult::ok(matches, 150);
        assert!(result.is_success());
        assert_eq!(result.matches.len(), 1);
        assert!(!result.truncated);
        assert!(result.summary_line().contains("[OK]"));
        assert!(result.summary_line().contains("1 matches"));
    }

    #[test]
    fn test_result_truncation() {
        let matches: Vec<AstGrepMatch> = (0..50)
            .map(|i| AstGrepMatch {
                file: format!("file{}.rs", i),
                line: i as u32,
                column: 1,
                end_line: i as u32,
                end_column: 10,
                text: format!("match {}", i),
                rule_id: None,
                severity: None,
                message: None,
            })
            .collect();

        let mut result = AstGrepResult::ok(matches, 200);
        result.truncate_to(10);
        assert!(result.truncated);
        assert_eq!(result.matches.len(), 10);
        assert_eq!(result.total_matches, 50);
        assert!(result.summary_line().contains("truncated from 50"));
    }

    #[test]
    fn test_result_timeout() {
        let result = AstGrepResult::timeout(10000);
        assert!(!result.is_success());
        assert!(result.timed_out);
        assert!(result.summary_line().contains("ERROR"));
    }

    #[test]
    fn test_result_error() {
        let result = AstGrepResult::err("binary not found", 0);
        assert!(!result.is_success());
        assert!(result.summary_line().contains("binary not found"));
    }

    #[test]
    fn test_match_display() {
        let m = AstGrepMatch {
            file: "src/lib.rs".to_string(),
            line: 42,
            column: 5,
            end_line: 42,
            end_column: 15,
            text: "foo.unwrap()".to_string(),
            rule_id: None,
            severity: None,
            message: None,
        };
        assert_eq!(m.to_string(), "src/lib.rs:42:5: foo.unwrap()");
    }

    #[test]
    fn test_validate_query() {
        let runner = AstGrepRunner::new();

        // Valid pattern query
        let q = AstGrepQuery::pattern("$X", "rust");
        assert!(runner.validate_query(&q).is_ok());

        // Valid rule query
        let q = AstGrepQuery::rule("no-unwrap", "rust");
        assert!(runner.validate_query(&q).is_ok());

        // Empty pattern and no rule
        let q = AstGrepQuery {
            pattern: String::new(),
            language: "rust".to_string(),
            paths: vec![],
            rule_id: None,
            json_output: true,
        };
        assert!(runner.validate_query(&q).is_err());

        // Empty language
        let q = AstGrepQuery::pattern("$X", "");
        assert!(runner.validate_query(&q).is_err());
    }

    #[test]
    fn test_apply_bounds() {
        let runner = AstGrepRunner::with_config(AstGrepConfig {
            max_matches: 5,
            ..Default::default()
        });

        let matches: Vec<AstGrepMatch> = (0..20)
            .map(|i| AstGrepMatch {
                file: format!("file{}.rs", i),
                line: i,
                column: 1,
                end_line: i,
                end_column: 10,
                text: format!("match {}", i),
                rule_id: None,
                severity: None,
                message: None,
            })
            .collect();

        let result = AstGrepResult::ok(matches, 100);
        let bounded = runner.apply_bounds(result);
        assert_eq!(bounded.matches.len(), 5);
        assert!(bounded.truncated);
    }

    #[test]
    fn test_config_serde() {
        let config = AstGrepConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AstGrepConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.binary, "sg");
        assert_eq!(parsed.timeout_ms, 10_000);
    }

    #[test]
    fn test_query_serde() {
        let query = AstGrepQuery::pattern("$X.unwrap()", "rust");
        let json = serde_json::to_string(&query).unwrap();
        let parsed: AstGrepQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pattern, "$X.unwrap()");
        assert_eq!(parsed.language, "rust");
    }
}
