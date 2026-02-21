//! Skill library: indexes successful patterns from retrospectives and
//! injects matching skills into work packets as hints.
//!
//! Skills are keyed by trigger context (error categories + file patterns +
//! optional task type). When a new task matches a skill's trigger, the
//! approach is injected into the work packet's `skill_hints` field.

use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::{AnalyticsError, AnalyticsResult};

use crate::feedback::error_parser::ErrorCategory;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A learned skill — an approach that worked for a given trigger context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier for this skill.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// When this skill should activate.
    pub trigger: SkillTrigger,
    /// The approach that worked (free-form text injected as a hint).
    pub approach: String,
    /// Number of times this skill led to a successful resolution.
    pub success_count: u32,
    /// Number of times this skill was tried but didn't help.
    pub failure_count: u32,
}

impl Skill {
    /// Confidence score: success / (success + failure), with minimum sample guard.
    ///
    /// Returns 0.0 if total samples < `min_samples`.
    pub fn confidence(&self, min_samples: u32) -> f64 {
        let total = self.success_count + self.failure_count;
        if total < min_samples {
            return 0.0;
        }
        f64::from(self.success_count) / f64::from(total)
    }

    /// Record a successful use of this skill.
    pub fn record_success(&mut self) {
        self.success_count += 1;
    }

    /// Record a failed use of this skill.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }
}

/// Trigger conditions that determine when a skill activates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTrigger {
    /// Error categories that must be present (any match counts).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub error_categories: Vec<ErrorCategory>,
    /// File path glob patterns (e.g., `"src/agents/*.rs"`).
    /// Any matching file in the task context triggers the skill.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_patterns: Vec<String>,
    /// Optional task type filter (e.g., `"bug"`, `"feature"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
}

impl SkillTrigger {
    /// Check if this trigger matches the given task context.
    ///
    /// Matching rules:
    /// - If `error_categories` is non-empty, at least one must match.
    /// - If `file_patterns` is non-empty, at least one context file must match a pattern.
    /// - If `task_type` is set, it must match (case-insensitive).
    /// - If all trigger fields are empty, it never matches (no wildcards).
    pub fn matches(&self, context: &TaskContext) -> bool {
        let has_any_condition = !self.error_categories.is_empty()
            || !self.file_patterns.is_empty()
            || self.task_type.is_some();

        if !has_any_condition {
            return false;
        }

        // Error category match (any-of)
        if !self.error_categories.is_empty() {
            let cat_match = self
                .error_categories
                .iter()
                .any(|tc| context.error_categories.contains(tc));
            if !cat_match {
                return false;
            }
        }

        // File pattern match (any file matches any pattern)
        if !self.file_patterns.is_empty() {
            let file_match = self.file_patterns.iter().any(|pattern| {
                context
                    .files_involved
                    .iter()
                    .any(|f| glob_matches(pattern, f))
            });
            if !file_match {
                return false;
            }
        }

        // Task type match (case-insensitive)
        if let Some(ref tt) = self.task_type {
            if let Some(ref ctx_tt) = context.task_type {
                if !tt.eq_ignore_ascii_case(ctx_tt) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// Context about the current task, used for skill matching.
#[derive(Debug, Clone, Default)]
pub struct TaskContext {
    /// Error categories encountered so far.
    pub error_categories: Vec<ErrorCategory>,
    /// Files involved in the task.
    pub files_involved: Vec<String>,
    /// Task type (e.g., "bug", "feature").
    pub task_type: Option<String>,
}

/// A skill hint injected into a work packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillHint {
    /// ID of the originating skill.
    pub skill_id: String,
    /// Human-readable label.
    pub label: String,
    /// The approach to try.
    pub approach: String,
    /// Confidence score at time of injection.
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// SkillLibrary
// ---------------------------------------------------------------------------

/// Minimum samples before a skill is considered for injection.
const DEFAULT_MIN_SAMPLES: u32 = 2;

/// Minimum confidence for a skill to be injected.
const DEFAULT_MIN_CONFIDENCE: f64 = 0.5;

/// Persistent skill library backed by a JSON file.
pub struct SkillLibrary {
    skills: Vec<Skill>,
    min_samples: u32,
    min_confidence: f64,
}

impl SkillLibrary {
    /// Create an empty skill library.
    pub fn new() -> Self {
        Self {
            skills: Vec::new(),
            min_samples: DEFAULT_MIN_SAMPLES,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }

    /// Load a skill library from a JSON file. Returns empty library if file doesn't exist.
    pub fn load(path: &Path) -> AnalyticsResult<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read_to_string(path).map_err(|e| AnalyticsError::FileRead {
            path: path.to_path_buf(),
            source: e,
        })?;
        let skills: Vec<Skill> = serde_json::from_str(&data)?;
        Ok(Self {
            skills,
            min_samples: DEFAULT_MIN_SAMPLES,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        })
    }

    /// Persist the skill library to a JSON file.
    pub fn save(&self, path: &Path) -> AnalyticsResult<()> {
        let data = serde_json::to_string_pretty(&self.skills)?;
        std::fs::write(path, data).map_err(|e| AnalyticsError::FileWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Override the minimum sample threshold.
    pub fn with_min_samples(mut self, min: u32) -> Self {
        self.min_samples = min;
        self
    }

    /// Override the minimum confidence threshold.
    pub fn with_min_confidence(mut self, min: f64) -> Self {
        self.min_confidence = min;
        self
    }

    /// Add a new skill to the library.
    pub fn add_skill(&mut self, skill: Skill) {
        self.skills.push(skill);
    }

    /// Create and add a new skill with a generated ID.
    pub fn create_skill(&mut self, label: &str, trigger: SkillTrigger, approach: &str) -> String {
        let id = format!("skill-{}", &Uuid::new_v4().to_string()[..8]);
        let skill = Skill {
            id: id.clone(),
            label: label.to_string(),
            trigger,
            approach: approach.to_string(),
            success_count: 1, // Created from a successful resolution
            failure_count: 0,
        };
        self.skills.push(skill);
        id
    }

    /// Find all skills matching the given context, ranked by confidence (descending).
    ///
    /// Only returns skills with sufficient samples and above the minimum confidence threshold.
    pub fn find_matching(&self, context: &TaskContext) -> Vec<SkillHint> {
        let mut hints: Vec<SkillHint> = self
            .skills
            .iter()
            .filter(|s| s.trigger.matches(context))
            .filter_map(|s| {
                let conf = s.confidence(self.min_samples);
                if conf >= self.min_confidence {
                    Some(SkillHint {
                        skill_id: s.id.clone(),
                        label: s.label.clone(),
                        approach: s.approach.clone(),
                        confidence: conf,
                    })
                } else {
                    None
                }
            })
            .collect();

        hints.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hints
    }

    /// Look up a skill by ID and record success.
    pub fn record_success(&mut self, skill_id: &str) {
        if let Some(skill) = self.skills.iter_mut().find(|s| s.id == skill_id) {
            skill.record_success();
        }
    }

    /// Look up a skill by ID and record failure.
    pub fn record_failure(&mut self, skill_id: &str) {
        if let Some(skill) = self.skills.iter_mut().find(|s| s.id == skill_id) {
            skill.record_failure();
        }
    }

    /// Get the number of skills in the library.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Get a reference to all skills.
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }
}

impl Default for SkillLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Glob matching (simple, no external dependency)
// ---------------------------------------------------------------------------

/// Simple glob matching supporting `*` (non-separator chars) and `?` (single non-separator char).
///
/// Used for file pattern matching in skill triggers. `*` does NOT match `/`
/// (path separator), so `src/*.rs` matches `src/lib.rs` but not `src/nested/lib.rs`.
fn glob_matches(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    glob_match_recursive(&pattern_chars, &text_chars, 0, 0)
}

fn glob_match_recursive(pattern: &[char], text: &[char], pi: usize, ti: usize) -> bool {
    if pi == pattern.len() {
        return ti == text.len();
    }

    match pattern[pi] {
        '*' => {
            // Match zero or more non-separator characters
            for skip in 0..=(text.len() - ti) {
                // Stop at path separators — `*` doesn't cross directories
                if skip > 0 && text[ti + skip - 1] == '/' {
                    break;
                }
                if glob_match_recursive(pattern, text, pi + 1, ti + skip) {
                    return true;
                }
            }
            false
        }
        '?' => {
            // Match exactly one non-separator character
            if ti < text.len() && text[ti] != '/' {
                glob_match_recursive(pattern, text, pi + 1, ti + 1)
            } else {
                false
            }
        }
        c => {
            if ti < text.len() && text[ti] == c {
                glob_match_recursive(pattern, text, pi + 1, ti + 1)
            } else {
                false
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trigger() -> SkillTrigger {
        SkillTrigger {
            error_categories: vec![ErrorCategory::BorrowChecker],
            file_patterns: vec!["src/*.rs".to_string()],
            task_type: None,
        }
    }

    fn sample_skill(label: &str, success: u32, failure: u32) -> Skill {
        Skill {
            id: format!("skill-{label}"),
            label: label.to_string(),
            trigger: sample_trigger(),
            approach: format!("Use the {label} approach"),
            success_count: success,
            failure_count: failure,
        }
    }

    // --- Glob matching ---

    #[test]
    fn test_glob_exact_match() {
        assert!(glob_matches("src/lib.rs", "src/lib.rs"));
    }

    #[test]
    fn test_glob_star_middle() {
        assert!(glob_matches("src/*.rs", "src/lib.rs"));
        assert!(glob_matches("src/*.rs", "src/main.rs"));
        assert!(!glob_matches("src/*.rs", "src/nested/lib.rs"));
    }

    #[test]
    fn test_glob_star_prefix() {
        assert!(glob_matches("*.rs", "lib.rs"));
        assert!(!glob_matches("*.rs", "src/lib.rs"));
    }

    #[test]
    fn test_glob_question_mark() {
        assert!(glob_matches("src/?.rs", "src/a.rs"));
        assert!(!glob_matches("src/?.rs", "src/ab.rs"));
    }

    #[test]
    fn test_glob_no_match() {
        assert!(!glob_matches("src/*.rs", "tests/test.rs"));
    }

    // --- Skill confidence ---

    #[test]
    fn test_confidence_below_min_samples() {
        let skill = sample_skill("a", 1, 0);
        assert_eq!(skill.confidence(2), 0.0);
    }

    #[test]
    fn test_confidence_at_min_samples() {
        let skill = sample_skill("a", 2, 0);
        assert_eq!(skill.confidence(2), 1.0);
    }

    #[test]
    fn test_confidence_mixed() {
        let skill = sample_skill("a", 3, 1);
        assert!((skill.confidence(2) - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_all_failures() {
        let skill = sample_skill("a", 0, 5);
        assert_eq!(skill.confidence(2), 0.0);
    }

    #[test]
    fn test_record_success_failure() {
        let mut skill = sample_skill("a", 2, 1);
        skill.record_success();
        assert_eq!(skill.success_count, 3);
        skill.record_failure();
        assert_eq!(skill.failure_count, 2);
    }

    // --- SkillTrigger matching ---

    #[test]
    fn test_trigger_matches_error_category() {
        let trigger = SkillTrigger {
            error_categories: vec![ErrorCategory::BorrowChecker, ErrorCategory::Lifetime],
            file_patterns: vec![],
            task_type: None,
        };
        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec![],
            task_type: None,
        };
        assert!(trigger.matches(&ctx));
    }

    #[test]
    fn test_trigger_no_match_wrong_category() {
        let trigger = SkillTrigger {
            error_categories: vec![ErrorCategory::BorrowChecker],
            file_patterns: vec![],
            task_type: None,
        };
        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::Lifetime],
            files_involved: vec![],
            task_type: None,
        };
        assert!(!trigger.matches(&ctx));
    }

    #[test]
    fn test_trigger_matches_file_pattern() {
        let trigger = SkillTrigger {
            error_categories: vec![],
            file_patterns: vec!["src/agents/*.rs".to_string()],
            task_type: None,
        };
        let ctx = TaskContext {
            error_categories: vec![],
            files_involved: vec!["src/agents/coder.rs".to_string()],
            task_type: None,
        };
        assert!(trigger.matches(&ctx));
    }

    #[test]
    fn test_trigger_no_match_wrong_file() {
        let trigger = SkillTrigger {
            error_categories: vec![],
            file_patterns: vec!["src/agents/*.rs".to_string()],
            task_type: None,
        };
        let ctx = TaskContext {
            error_categories: vec![],
            files_involved: vec!["tests/test.rs".to_string()],
            task_type: None,
        };
        assert!(!trigger.matches(&ctx));
    }

    #[test]
    fn test_trigger_matches_task_type() {
        let trigger = SkillTrigger {
            error_categories: vec![],
            file_patterns: vec![],
            task_type: Some("bug".to_string()),
        };
        let ctx = TaskContext {
            error_categories: vec![],
            files_involved: vec![],
            task_type: Some("Bug".to_string()),
        };
        assert!(trigger.matches(&ctx));
    }

    #[test]
    fn test_trigger_combined_all_must_match() {
        let trigger = SkillTrigger {
            error_categories: vec![ErrorCategory::BorrowChecker],
            file_patterns: vec!["src/*.rs".to_string()],
            task_type: Some("bug".to_string()),
        };

        // All match
        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/lib.rs".to_string()],
            task_type: Some("bug".to_string()),
        };
        assert!(trigger.matches(&ctx));

        // Missing file match
        let ctx2 = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["tests/test.rs".to_string()],
            task_type: Some("bug".to_string()),
        };
        assert!(!trigger.matches(&ctx2));
    }

    #[test]
    fn test_trigger_empty_never_matches() {
        let trigger = SkillTrigger {
            error_categories: vec![],
            file_patterns: vec![],
            task_type: None,
        };
        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/lib.rs".to_string()],
            task_type: Some("bug".to_string()),
        };
        assert!(!trigger.matches(&ctx));
    }

    // --- SkillLibrary ---

    #[test]
    fn test_library_add_and_find() {
        let mut lib = SkillLibrary::new().with_min_samples(1);
        lib.add_skill(sample_skill("borrow-fix", 3, 1));

        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/main.rs".to_string()],
            task_type: None,
        };

        let hints = lib.find_matching(&ctx);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].skill_id, "skill-borrow-fix");
        assert!((hints[0].confidence - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_library_filters_low_confidence() {
        let mut lib = SkillLibrary::new()
            .with_min_samples(1)
            .with_min_confidence(0.8);
        lib.add_skill(sample_skill("low-conf", 3, 2)); // 60% < 80% threshold

        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/lib.rs".to_string()],
            task_type: None,
        };

        let hints = lib.find_matching(&ctx);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_library_ranked_by_confidence() {
        let mut lib = SkillLibrary::new().with_min_samples(1);
        lib.add_skill(sample_skill("low", 2, 2)); // 50%
        lib.add_skill(sample_skill("high", 4, 1)); // 80%
        lib.add_skill(sample_skill("mid", 3, 1)); // 75%

        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/lib.rs".to_string()],
            task_type: None,
        };

        let hints = lib.find_matching(&ctx);
        assert_eq!(hints.len(), 3);
        assert_eq!(hints[0].skill_id, "skill-high");
        assert_eq!(hints[1].skill_id, "skill-mid");
        assert_eq!(hints[2].skill_id, "skill-low");
    }

    #[test]
    fn test_library_create_skill() {
        let mut lib = SkillLibrary::new();
        let id = lib.create_skill(
            "Arc pattern",
            SkillTrigger {
                error_categories: vec![ErrorCategory::BorrowChecker],
                file_patterns: vec![],
                task_type: None,
            },
            "Wrap shared state in Arc<Mutex<>>",
        );

        assert!(id.starts_with("skill-"));
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.skills()[0].success_count, 1);
    }

    #[test]
    fn test_library_record_outcome() {
        let mut lib = SkillLibrary::new();
        lib.add_skill(sample_skill("test", 2, 0));

        lib.record_success("skill-test");
        assert_eq!(lib.skills()[0].success_count, 3);

        lib.record_failure("skill-test");
        assert_eq!(lib.skills()[0].failure_count, 1);

        // Unknown ID is a no-op
        lib.record_success("nonexistent");
    }

    #[test]
    fn test_library_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("skills.json");

        let mut lib = SkillLibrary::new();
        lib.add_skill(sample_skill("persisted", 5, 1));
        lib.save(&path).unwrap();

        let loaded = SkillLibrary::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.skills()[0].id, "skill-persisted");
        assert_eq!(loaded.skills()[0].success_count, 5);
    }

    #[test]
    fn test_library_load_nonexistent_returns_empty() {
        let lib = SkillLibrary::load(Path::new("/nonexistent/skills.json")).unwrap();
        assert!(lib.is_empty());
    }

    #[test]
    fn test_skill_serde_roundtrip() {
        let skill = sample_skill("serde-test", 3, 1);
        let json = serde_json::to_string(&skill).unwrap();
        let deserialized: Skill = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, skill.id);
        assert_eq!(deserialized.success_count, 3);
        assert_eq!(deserialized.failure_count, 1);
    }

    #[test]
    fn test_skill_hint_serde_roundtrip() {
        let hint = SkillHint {
            skill_id: "skill-001".into(),
            label: "Test hint".into(),
            approach: "Do something".into(),
            confidence: 0.85,
        };
        let json = serde_json::to_string(&hint).unwrap();
        let deserialized: SkillHint = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.skill_id, "skill-001");
        assert!((deserialized.confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_library_find_returns_empty() {
        let lib = SkillLibrary::new();
        let ctx = TaskContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            files_involved: vec!["src/lib.rs".into()],
            task_type: None,
        };
        assert!(lib.find_matching(&ctx).is_empty());
    }
}
