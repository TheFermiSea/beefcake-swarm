//! Rule pack mapping — maps anti-pattern checks to ast-grep rules and sgconfig.
//!
//! Provides a registry of rule packs that can be selected per-language and
//! per-severity, mapping to concrete ast-grep rule IDs and sgconfig paths.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity level for a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RuleSeverity {
    /// Must fix — blocks merge.
    Error,
    /// Should fix — reviewer flags.
    Warning,
    /// Nice to fix — informational.
    Info,
}

impl std::fmt::Display for RuleSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleSeverity::Error => write!(f, "error"),
            RuleSeverity::Warning => write!(f, "warning"),
            RuleSeverity::Info => write!(f, "info"),
        }
    }
}

/// A single rule entry in a rule pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePackEntry {
    /// Rule ID matching ast-grep rule configuration.
    pub rule_id: String,
    /// Human-readable description of what the rule catches.
    pub description: String,
    /// Severity of violations.
    pub severity: RuleSeverity,
    /// Language this rule applies to.
    pub language: String,
    /// The ast-grep pattern (if pattern-based rather than rule-file-based).
    pub pattern: Option<String>,
    /// Path to the sgconfig rule file (if rule-file-based).
    pub rule_file: Option<String>,
    /// Category tag for grouping (e.g., "safety", "performance", "style").
    pub category: String,
    /// Whether this rule is enabled by default.
    pub enabled: bool,
}

impl RulePackEntry {
    /// Create a pattern-based rule.
    pub fn pattern_rule(
        rule_id: &str,
        description: &str,
        severity: RuleSeverity,
        language: &str,
        pattern: &str,
        category: &str,
    ) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            description: description.to_string(),
            severity,
            language: language.to_string(),
            pattern: Some(pattern.to_string()),
            rule_file: None,
            category: category.to_string(),
            enabled: true,
        }
    }

    /// Create a rule-file-based rule.
    pub fn file_rule(
        rule_id: &str,
        description: &str,
        severity: RuleSeverity,
        language: &str,
        rule_file: &str,
        category: &str,
    ) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            description: description.to_string(),
            severity,
            language: language.to_string(),
            pattern: None,
            rule_file: Some(rule_file.to_string()),
            category: category.to_string(),
            enabled: true,
        }
    }

    /// Set the enabled state.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// A named collection of rules for a specific purpose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePack {
    /// Pack name (e.g., "rust-safety", "rust-performance").
    pub name: String,
    /// Description of the pack's purpose.
    pub description: String,
    /// Rules in this pack.
    pub rules: Vec<RulePackEntry>,
}

impl RulePack {
    /// Create a new empty rule pack.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            rules: Vec::new(),
        }
    }

    /// Add a rule to the pack.
    pub fn add_rule(&mut self, rule: RulePackEntry) {
        self.rules.push(rule);
    }

    /// Get all enabled rules.
    pub fn enabled_rules(&self) -> Vec<&RulePackEntry> {
        self.rules.iter().filter(|r| r.enabled).collect()
    }

    /// Get rules filtered by severity.
    pub fn by_severity(&self, severity: RuleSeverity) -> Vec<&RulePackEntry> {
        self.rules
            .iter()
            .filter(|r| r.enabled && r.severity == severity)
            .collect()
    }

    /// Get rules filtered by category.
    pub fn by_category(&self, category: &str) -> Vec<&RulePackEntry> {
        self.rules
            .iter()
            .filter(|r| r.enabled && r.category == category)
            .collect()
    }

    /// Get rules for a specific language.
    pub fn for_language(&self, language: &str) -> Vec<&RulePackEntry> {
        self.rules
            .iter()
            .filter(|r| r.enabled && r.language == language)
            .collect()
    }

    /// Count of enabled rules.
    pub fn enabled_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }
}

/// Registry of all available rule packs.
pub struct RulePackRegistry {
    packs: HashMap<String, RulePack>,
}

impl RulePackRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            packs: HashMap::new(),
        }
    }

    /// Create a registry pre-loaded with the default Rust rule packs.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Self::rust_safety_pack());
        registry.register(Self::rust_performance_pack());
        registry.register(Self::rust_style_pack());
        registry
    }

    /// Register a rule pack.
    pub fn register(&mut self, pack: RulePack) {
        self.packs.insert(pack.name.clone(), pack);
    }

    /// Get a pack by name.
    pub fn get(&self, name: &str) -> Option<&RulePack> {
        self.packs.get(name)
    }

    /// List all registered pack names.
    pub fn pack_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.packs.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Get all enabled rules across all packs for a language.
    pub fn all_rules_for_language(&self, language: &str) -> Vec<&RulePackEntry> {
        self.packs
            .values()
            .flat_map(|p| p.for_language(language))
            .collect()
    }

    /// Get all error-severity rules across all packs.
    pub fn blocking_rules(&self) -> Vec<&RulePackEntry> {
        self.packs
            .values()
            .flat_map(|p| p.by_severity(RuleSeverity::Error))
            .collect()
    }

    /// Total enabled rules across all packs.
    pub fn total_enabled(&self) -> usize {
        self.packs.values().map(|p| p.enabled_count()).sum()
    }

    /// Default Rust safety rule pack.
    fn rust_safety_pack() -> RulePack {
        let mut pack = RulePack::new("rust-safety", "Rust safety and correctness checks");
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-unwrap",
            "Avoid .unwrap() — use ? or explicit error handling",
            RuleSeverity::Error,
            "rust",
            "$EXPR.unwrap()",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-expect",
            "Avoid .expect() in library code — use ? or explicit error handling",
            RuleSeverity::Warning,
            "rust",
            "$EXPR.expect($MSG)",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-unsafe",
            "Unsafe blocks require justification comment",
            RuleSeverity::Error,
            "rust",
            "unsafe { $$$BODY }",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-panic",
            "Avoid panic!() — use Result return types",
            RuleSeverity::Error,
            "rust",
            "panic!($$$ARGS)",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-todo",
            "Remove TODO macros before merge",
            RuleSeverity::Warning,
            "rust",
            "todo!($$$ARGS)",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-unimplemented",
            "Remove unimplemented!() before merge",
            RuleSeverity::Error,
            "rust",
            "unimplemented!($$$ARGS)",
            "safety",
        ));
        pack
    }

    /// Default Rust performance rule pack.
    fn rust_performance_pack() -> RulePack {
        let mut pack = RulePack::new("rust-performance", "Rust performance anti-patterns");
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-clone-in-loop",
            "Avoid .clone() inside loops — consider borrowing",
            RuleSeverity::Warning,
            "rust",
            "$EXPR.clone()",
            "performance",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-collect-iter",
            "Avoid .collect::<Vec<_>>() followed by .iter() — chain iterators instead",
            RuleSeverity::Info,
            "rust",
            "$EXPR.collect::<Vec<$T>>()",
            "performance",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-format-in-loop",
            "Avoid format!() in hot loops — preallocate or use write!",
            RuleSeverity::Info,
            "rust",
            "format!($$$ARGS)",
            "performance",
        ));
        pack
    }

    /// Default Rust style rule pack.
    fn rust_style_pack() -> RulePack {
        let mut pack = RulePack::new("rust-style", "Rust style and idiomatic patterns");
        pack.add_rule(RulePackEntry::pattern_rule(
            "use-if-let",
            "Prefer if let over match with single arm and wildcard",
            RuleSeverity::Info,
            "rust",
            "match $EXPR { $PAT => $BODY, _ => {} }",
            "style",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "no-string-to-string",
            "Use .to_owned() or String::from() instead of .to_string() for &str",
            RuleSeverity::Info,
            "rust",
            "$EXPR.to_string()",
            "style",
        ));
        pack
    }
}

impl RulePackRegistry {
    /// Load a single rule pack from a YAML string.
    ///
    /// The YAML format matches the `RulePack` serde structure:
    /// ```yaml
    /// name: my-pack
    /// description: My custom rules
    /// rules:
    ///   - rule_id: no-debug
    ///     description: Remove debug prints
    ///     severity: Warning
    ///     language: rust
    ///     pattern: "dbg!($$$ARGS)"
    ///     category: cleanup
    ///     enabled: true
    /// ```
    pub fn load_yaml(yaml: &str) -> Result<RulePack, RuleIngestionError> {
        serde_yaml::from_str(yaml).map_err(|e| RuleIngestionError {
            source_path: None,
            kind: IngestionErrorKind::ParseError,
            detail: format!("YAML parse error: {}", e),
        })
    }

    /// Load a rule pack from a YAML file path.
    pub fn load_yaml_file(path: &std::path::Path) -> Result<RulePack, RuleIngestionError> {
        let content = std::fs::read_to_string(path).map_err(|e| RuleIngestionError {
            source_path: Some(path.display().to_string()),
            kind: IngestionErrorKind::IoError,
            detail: format!("Failed to read file: {}", e),
        })?;
        let mut pack =
            serde_yaml::from_str::<RulePack>(&content).map_err(|e| RuleIngestionError {
                source_path: Some(path.display().to_string()),
                kind: IngestionErrorKind::ParseError,
                detail: format!("YAML parse error: {}", e),
            })?;

        // Tag rules with source file for traceability
        for rule in &mut pack.rules {
            if rule.rule_file.is_none() {
                rule.rule_file = Some(path.display().to_string());
            }
        }

        Ok(pack)
    }

    /// Load all YAML rule pack files from a directory.
    ///
    /// Scans for `*.yml` and `*.yaml` files, loads each as a `RulePack`,
    /// and registers them. Returns a summary of loaded packs and errors.
    pub fn load_directory(
        &mut self,
        dir: &std::path::Path,
    ) -> Result<IngestionSummary, RuleIngestionError> {
        if !dir.is_dir() {
            return Err(RuleIngestionError {
                source_path: Some(dir.display().to_string()),
                kind: IngestionErrorKind::IoError,
                detail: "Path is not a directory".to_string(),
            });
        }

        let mut summary = IngestionSummary::default();

        let entries = std::fs::read_dir(dir).map_err(|e| RuleIngestionError {
            source_path: Some(dir.display().to_string()),
            kind: IngestionErrorKind::IoError,
            detail: format!("Failed to read directory: {}", e),
        })?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    summary.errors.push(RuleIngestionError {
                        source_path: None,
                        kind: IngestionErrorKind::IoError,
                        detail: format!("Failed to read dir entry: {}", e),
                    });
                    continue;
                }
            };

            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "yml" && ext != "yaml" {
                continue;
            }

            match Self::load_yaml_file(&path) {
                Ok(pack) => {
                    let name = pack.name.clone();
                    let rule_count = pack.enabled_count();
                    self.register(pack);
                    summary.loaded_packs.push(name);
                    summary.total_rules += rule_count;
                }
                Err(e) => {
                    summary.errors.push(e);
                }
            }
        }

        summary.loaded_packs.sort();
        Ok(summary)
    }

    /// Create a registry with defaults, then overlay packs from a directory.
    pub fn with_defaults_and_directory(
        dir: &std::path::Path,
    ) -> Result<(Self, IngestionSummary), RuleIngestionError> {
        let mut registry = Self::with_defaults();
        let summary = registry.load_directory(dir)?;
        Ok((registry, summary))
    }

    /// Export a pack to YAML string.
    pub fn export_yaml(pack: &RulePack) -> Result<String, RuleIngestionError> {
        serde_yaml::to_string(pack).map_err(|e| RuleIngestionError {
            source_path: None,
            kind: IngestionErrorKind::SerializeError,
            detail: format!("YAML serialize error: {}", e),
        })
    }

    /// Compute a version hash for all registered rules (deterministic ordering).
    pub fn version_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut sorted_packs: Vec<_> = self.packs.iter().collect();
        sorted_packs.sort_by_key(|(name, _)| (*name).clone());
        for (name, pack) in sorted_packs {
            name.hash(&mut hasher);
            pack.rules.len().hash(&mut hasher);
            for rule in &pack.rules {
                rule.rule_id.hash(&mut hasher);
                rule.enabled.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

/// Error from rule ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleIngestionError {
    /// Path that caused the error (if file-based).
    pub source_path: Option<String>,
    /// Error kind.
    pub kind: IngestionErrorKind,
    /// Human-readable detail.
    pub detail: String,
}

impl std::fmt::Display for RuleIngestionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source_path {
            Some(path) => write!(f, "[{}] {}: {}", path, self.kind, self.detail),
            None => write!(f, "{}: {}", self.kind, self.detail),
        }
    }
}

impl std::error::Error for RuleIngestionError {}

/// Kind of ingestion error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestionErrorKind {
    /// File I/O error.
    IoError,
    /// YAML parse error.
    ParseError,
    /// YAML serialize error.
    SerializeError,
}

impl std::fmt::Display for IngestionErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError => write!(f, "io_error"),
            Self::ParseError => write!(f, "parse_error"),
            Self::SerializeError => write!(f, "serialize_error"),
        }
    }
}

/// Summary of a directory ingestion operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestionSummary {
    /// Names of successfully loaded packs.
    pub loaded_packs: Vec<String>,
    /// Total rules across loaded packs.
    pub total_rules: usize,
    /// Errors encountered during loading.
    pub errors: Vec<RuleIngestionError>,
}

impl IngestionSummary {
    /// Whether all files loaded successfully.
    pub fn all_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl Default for RulePackRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_display() {
        assert_eq!(RuleSeverity::Error.to_string(), "error");
        assert_eq!(RuleSeverity::Warning.to_string(), "warning");
        assert_eq!(RuleSeverity::Info.to_string(), "info");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(RuleSeverity::Error < RuleSeverity::Warning);
        assert!(RuleSeverity::Warning < RuleSeverity::Info);
    }

    #[test]
    fn test_pattern_rule() {
        let rule = RulePackEntry::pattern_rule(
            "test-rule",
            "Test description",
            RuleSeverity::Warning,
            "rust",
            "$X.unwrap()",
            "safety",
        );
        assert_eq!(rule.rule_id, "test-rule");
        assert_eq!(rule.pattern.as_deref(), Some("$X.unwrap()"));
        assert!(rule.rule_file.is_none());
        assert!(rule.enabled);
    }

    #[test]
    fn test_file_rule() {
        let rule = RulePackEntry::file_rule(
            "custom-rule",
            "Custom check",
            RuleSeverity::Error,
            "rust",
            "rules/custom.yml",
            "custom",
        );
        assert!(rule.pattern.is_none());
        assert_eq!(rule.rule_file.as_deref(), Some("rules/custom.yml"));
    }

    #[test]
    fn test_rule_disabled() {
        let rule = RulePackEntry::pattern_rule(
            "disabled-rule",
            "Should be off",
            RuleSeverity::Info,
            "rust",
            "$X",
            "test",
        )
        .with_enabled(false);
        assert!(!rule.enabled);
    }

    #[test]
    fn test_rule_pack_filtering() {
        let mut pack = RulePack::new("test", "Test pack");
        pack.add_rule(RulePackEntry::pattern_rule(
            "err1",
            "Error rule",
            RuleSeverity::Error,
            "rust",
            "$X",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "warn1",
            "Warning rule",
            RuleSeverity::Warning,
            "rust",
            "$Y",
            "performance",
        ));
        pack.add_rule(
            RulePackEntry::pattern_rule(
                "info1",
                "Info rule (disabled)",
                RuleSeverity::Info,
                "rust",
                "$Z",
                "safety",
            )
            .with_enabled(false),
        );

        assert_eq!(pack.enabled_count(), 2);
        assert_eq!(pack.enabled_rules().len(), 2);
        assert_eq!(pack.by_severity(RuleSeverity::Error).len(), 1);
        assert_eq!(pack.by_severity(RuleSeverity::Warning).len(), 1);
        assert_eq!(pack.by_severity(RuleSeverity::Info).len(), 0); // disabled
        assert_eq!(pack.by_category("safety").len(), 1); // one enabled safety
        assert_eq!(pack.by_category("performance").len(), 1);
    }

    #[test]
    fn test_rule_pack_language_filter() {
        let mut pack = RulePack::new("multi", "Multi-language pack");
        pack.add_rule(RulePackEntry::pattern_rule(
            "rs1",
            "Rust rule",
            RuleSeverity::Error,
            "rust",
            "$X",
            "safety",
        ));
        pack.add_rule(RulePackEntry::pattern_rule(
            "ts1",
            "TypeScript rule",
            RuleSeverity::Error,
            "typescript",
            "$Y",
            "safety",
        ));

        assert_eq!(pack.for_language("rust").len(), 1);
        assert_eq!(pack.for_language("typescript").len(), 1);
        assert_eq!(pack.for_language("python").len(), 0);
    }

    #[test]
    fn test_default_registry() {
        let registry = RulePackRegistry::with_defaults();
        let names = registry.pack_names();
        assert!(names.contains(&"rust-safety"));
        assert!(names.contains(&"rust-performance"));
        assert!(names.contains(&"rust-style"));
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_registry_rust_rules() {
        let registry = RulePackRegistry::with_defaults();
        let rust_rules = registry.all_rules_for_language("rust");
        // 6 safety + 3 performance + 2 style = 11
        assert_eq!(rust_rules.len(), 11);
    }

    #[test]
    fn test_registry_blocking_rules() {
        let registry = RulePackRegistry::with_defaults();
        let blockers = registry.blocking_rules();
        // no-unwrap, no-unsafe, no-panic, no-unimplemented = 4 errors
        assert_eq!(blockers.len(), 4);
        for rule in &blockers {
            assert_eq!(rule.severity, RuleSeverity::Error);
        }
    }

    #[test]
    fn test_registry_total_enabled() {
        let registry = RulePackRegistry::with_defaults();
        assert_eq!(registry.total_enabled(), 11);
    }

    #[test]
    fn test_registry_get_pack() {
        let registry = RulePackRegistry::with_defaults();
        let safety = registry.get("rust-safety").unwrap();
        assert_eq!(safety.name, "rust-safety");
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_custom_registry() {
        let mut registry = RulePackRegistry::new();
        assert_eq!(registry.total_enabled(), 0);

        let mut pack = RulePack::new("custom", "Custom pack");
        pack.add_rule(RulePackEntry::pattern_rule(
            "custom-1",
            "Custom rule",
            RuleSeverity::Warning,
            "python",
            "import $X",
            "imports",
        ));
        registry.register(pack);

        assert_eq!(registry.total_enabled(), 1);
        assert_eq!(registry.all_rules_for_language("python").len(), 1);
        assert_eq!(registry.all_rules_for_language("rust").len(), 0);
    }

    #[test]
    fn test_rule_pack_serde() {
        let registry = RulePackRegistry::with_defaults();
        let pack = registry.get("rust-safety").unwrap();
        let json = serde_json::to_string(pack).unwrap();
        let parsed: RulePack = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "rust-safety");
        assert_eq!(parsed.rules.len(), pack.rules.len());
    }

    #[test]
    fn test_entry_serde() {
        let rule = RulePackEntry::pattern_rule(
            "test",
            "Test",
            RuleSeverity::Error,
            "rust",
            "$X.unwrap()",
            "safety",
        );
        let json = serde_json::to_string(&rule).unwrap();
        let parsed: RulePackEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rule_id, "test");
        assert_eq!(parsed.severity, RuleSeverity::Error);
    }
}
