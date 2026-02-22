//! Telemetry-Driven Knowledge Base Refresh
//!
//! Periodically analyzes aggregate telemetry patterns and skill library
//! to maintain the NotebookLM knowledge base:
//!
//! - **Deprecate** stale patterns that haven't matched recent sessions
//! - **Promote** high-confidence skills to Project Brain documentation
//! - **Flag** frequent error categories that lack KB entries
//!
//! Each action is logged as a [`KBRefreshAction`] for audit.

use std::collections::HashSet;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use coordination::analytics::skills::SkillLibrary;

use crate::telemetry::AggregateAnalytics;

// ── Configuration ────────────────────────────────────────────────────

/// Policy controlling when and how KB refresh runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshPolicy {
    /// How many sessions between refresh cycles (default: 10).
    pub session_interval: u32,
    /// Days of inactivity before a skill is considered stale.
    pub staleness_threshold_days: u32,
    /// Minimum confidence for a skill to be promoted to documentation.
    pub min_skill_confidence: f64,
    /// Minimum occurrences for an error category to be flagged as undocumented.
    pub min_error_occurrences: usize,
    /// Error categories already covered in the KB (prevents re-flagging).
    pub documented_error_categories: HashSet<String>,
}

impl Default for RefreshPolicy {
    fn default() -> Self {
        Self {
            session_interval: 10,
            staleness_threshold_days: 30,
            min_skill_confidence: 0.75,
            min_error_occurrences: 5,
            documented_error_categories: HashSet::new(),
        }
    }
}

impl RefreshPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session_interval(mut self, interval: u32) -> Self {
        self.session_interval = interval;
        self
    }

    pub fn with_staleness_days(mut self, days: u32) -> Self {
        self.staleness_threshold_days = days;
        self
    }

    pub fn with_min_confidence(mut self, conf: f64) -> Self {
        self.min_skill_confidence = conf;
        self
    }

    pub fn with_min_error_occurrences(mut self, count: usize) -> Self {
        self.min_error_occurrences = count;
        self
    }

    pub fn with_documented_categories(mut self, cats: HashSet<String>) -> Self {
        self.documented_error_categories = cats;
        self
    }

    /// Staleness threshold as a std Duration.
    pub fn staleness_duration(&self) -> Duration {
        Duration::from_secs(u64::from(self.staleness_threshold_days) * 86400)
    }
}

// ── Refresh Actions ──────────────────────────────────────────────────

/// A single KB maintenance action recommended by the refresh analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KBRefreshAction {
    /// What kind of action to take.
    pub action_type: RefreshActionType,
    /// Human-readable description.
    pub description: String,
    /// Which notebook target is affected.
    pub target: String,
    /// Severity/priority of the action.
    pub priority: ActionPriority,
}

/// Types of refresh actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshActionType {
    /// Skill hasn't been used in recent sessions — mark as potentially outdated.
    DeprecateStalePattern,
    /// High-confidence skill should be promoted to prominent documentation.
    PromoteSkill,
    /// Frequently occurring error category has no KB documentation.
    FlagUndocumentedError,
}

/// Priority of a refresh action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPriority {
    Low,
    Medium,
    High,
}

impl std::fmt::Display for RefreshActionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeprecateStalePattern => write!(f, "deprecate_stale_pattern"),
            Self::PromoteSkill => write!(f, "promote_skill"),
            Self::FlagUndocumentedError => write!(f, "flag_undocumented_error"),
        }
    }
}

// ── Refresh Report ───────────────────────────────────────────────────

/// Summary of a KB refresh analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshReport {
    /// When the analysis was performed.
    pub analyzed_at: String,
    /// Actions recommended.
    pub actions: Vec<KBRefreshAction>,
    /// Total skills analyzed.
    pub skills_analyzed: usize,
    /// Total error categories analyzed.
    pub error_categories_analyzed: usize,
    /// Number of stale skills found.
    pub stale_skills: usize,
    /// Number of skills promoted.
    pub promotions: usize,
    /// Number of undocumented errors flagged.
    pub undocumented_errors: usize,
}

impl RefreshReport {
    /// Whether the analysis produced any actions.
    pub fn has_actions(&self) -> bool {
        !self.actions.is_empty()
    }

    /// Count of actions by type.
    pub fn action_count(&self, action_type: &RefreshActionType) -> usize {
        self.actions
            .iter()
            .filter(|a| &a.action_type == action_type)
            .count()
    }
}

impl std::fmt::Display for RefreshReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KB Refresh: {} actions (stale={}, promoted={}, undocumented={})",
            self.actions.len(),
            self.stale_skills,
            self.promotions,
            self.undocumented_errors,
        )
    }
}

// ── Service ──────────────────────────────────────────────────────────

/// Whether a refresh should run based on session count.
pub fn should_refresh(total_sessions: usize, policy: &RefreshPolicy) -> bool {
    if policy.session_interval == 0 {
        return false;
    }
    total_sessions > 0 && total_sessions.is_multiple_of(policy.session_interval as usize)
}

/// Run a KB refresh analysis.
///
/// Examines the skill library and aggregate analytics to produce a set
/// of [`KBRefreshAction`]s. The caller is responsible for executing
/// the actions (e.g., uploading to NotebookLM, marking skills as deprecated).
pub fn analyze_and_refresh(
    analytics: &AggregateAnalytics,
    skills: &SkillLibrary,
    policy: &RefreshPolicy,
    now: DateTime<Utc>,
) -> RefreshReport {
    let mut actions = Vec::new();

    // 1. Deprecate stale skills
    let stale = find_stale_skills(skills, policy);
    let stale_count = stale.len();
    actions.extend(stale);

    // 2. Promote high-confidence skills
    let promoted = find_promotable_skills(skills, policy);
    let promotion_count = promoted.len();
    actions.extend(promoted);

    // 3. Flag undocumented error categories
    let undocumented = find_undocumented_errors(analytics, policy);
    let undocumented_count = undocumented.len();
    actions.extend(undocumented);

    RefreshReport {
        analyzed_at: now.to_rfc3339(),
        actions,
        skills_analyzed: skills.len(),
        error_categories_analyzed: analytics.error_category_frequencies.len(),
        stale_skills: stale_count,
        promotions: promotion_count,
        undocumented_errors: undocumented_count,
    }
}

// ── Internal Analysis Functions ──────────────────────────────────────

/// Find skills that are considered stale (low total usage).
///
/// A skill is stale if it has fewer than 2 total uses — it was created
/// once but never triggered again.
fn find_stale_skills(skills: &SkillLibrary, _policy: &RefreshPolicy) -> Vec<KBRefreshAction> {
    skills
        .skills()
        .iter()
        .filter(|s| {
            let total = s.success_count + s.failure_count;
            total <= 1 // Only the initial creation, never re-triggered
        })
        .map(|s| KBRefreshAction {
            action_type: RefreshActionType::DeprecateStalePattern,
            description: format!(
                "Skill '{}' (id={}) has only {} total use(s) — consider deprecating",
                s.label,
                s.id,
                s.success_count + s.failure_count,
            ),
            target: "debugging_kb".to_string(),
            priority: ActionPriority::Low,
        })
        .collect()
}

/// Find high-confidence skills that should be promoted to documentation.
fn find_promotable_skills(skills: &SkillLibrary, policy: &RefreshPolicy) -> Vec<KBRefreshAction> {
    let min_samples = 3; // Need meaningful sample size for promotion
    skills
        .skills()
        .iter()
        .filter(|s| {
            let conf = s.confidence(min_samples);
            conf >= policy.min_skill_confidence
        })
        .map(|s| KBRefreshAction {
            action_type: RefreshActionType::PromoteSkill,
            description: format!(
                "Skill '{}' (id={}) has {:.0}% confidence ({}/{} success) — promote to Project Brain",
                s.label,
                s.id,
                s.confidence(min_samples) * 100.0,
                s.success_count,
                s.success_count + s.failure_count,
            ),
            target: "project_brain".to_string(),
            priority: ActionPriority::Medium,
        })
        .collect()
}

/// Find error categories that occur frequently but lack documentation.
fn find_undocumented_errors(
    analytics: &AggregateAnalytics,
    policy: &RefreshPolicy,
) -> Vec<KBRefreshAction> {
    analytics
        .error_category_frequencies
        .iter()
        .filter(|(cat, count)| {
            **count >= policy.min_error_occurrences
                && !policy.documented_error_categories.contains(cat.as_str())
        })
        .map(|(cat, count)| KBRefreshAction {
            action_type: RefreshActionType::FlagUndocumentedError,
            description: format!(
                "Error category '{}' occurred {} times but has no KB documentation",
                cat, count,
            ),
            target: "debugging_kb".to_string(),
            priority: if *count >= policy.min_error_occurrences * 3 {
                ActionPriority::High
            } else {
                ActionPriority::Medium
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_analytics(error_freqs: HashMap<String, usize>) -> AggregateAnalytics {
        AggregateAnalytics {
            total_sessions: 20,
            success_rate: 0.75,
            average_iterations: 3.0,
            average_elapsed_ms: 15000.0,
            total_prompt_tokens: 50000,
            total_completion_tokens: 30000,
            error_category_frequencies: error_freqs,
        }
    }

    fn make_skill(label: &str, successes: u32, failures: u32) -> coordination::Skill {
        coordination::Skill {
            id: format!("skill-{}", label),
            label: label.to_string(),
            trigger: coordination::SkillTrigger {
                error_categories: vec![],
                file_patterns: vec![],
                task_type: None,
            },
            approach: format!("{} approach", label),
            success_count: successes,
            failure_count: failures,
        }
    }

    #[test]
    fn test_refresh_policy_defaults() {
        let policy = RefreshPolicy::default();
        assert_eq!(policy.session_interval, 10);
        assert_eq!(policy.staleness_threshold_days, 30);
        assert_eq!(policy.min_skill_confidence, 0.75);
        assert_eq!(policy.min_error_occurrences, 5);
    }

    #[test]
    fn test_should_refresh() {
        let policy = RefreshPolicy::new().with_session_interval(10);
        assert!(!should_refresh(0, &policy));
        assert!(!should_refresh(5, &policy));
        assert!(should_refresh(10, &policy));
        assert!(should_refresh(20, &policy));
        assert!(!should_refresh(15, &policy));
    }

    #[test]
    fn test_should_refresh_disabled() {
        let policy = RefreshPolicy::new().with_session_interval(0);
        assert!(!should_refresh(10, &policy));
        assert!(!should_refresh(100, &policy));
    }

    #[test]
    fn test_empty_analysis() {
        let analytics = make_analytics(HashMap::new());
        let skills = SkillLibrary::new();
        let policy = RefreshPolicy::default();
        let now = Utc::now();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert!(!report.has_actions());
        assert_eq!(report.skills_analyzed, 0);
        assert_eq!(report.stale_skills, 0);
        assert_eq!(report.promotions, 0);
        assert_eq!(report.undocumented_errors, 0);
    }

    #[test]
    fn test_detect_stale_skills() {
        let mut skills = SkillLibrary::new();
        skills.add_skill(make_skill("stale-one", 1, 0)); // Only initial creation
        skills.add_skill(make_skill("active", 8, 2)); // Actively used
        skills.add_skill(make_skill("stale-zero", 0, 0)); // Never triggered

        let analytics = make_analytics(HashMap::new());
        let policy = RefreshPolicy::default();
        let now = Utc::now();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert_eq!(report.stale_skills, 2); // stale-one and stale-zero
        assert_eq!(
            report.action_count(&RefreshActionType::DeprecateStalePattern),
            2
        );
    }

    #[test]
    fn test_promote_high_confidence_skills() {
        let mut skills = SkillLibrary::new();
        skills.add_skill(make_skill("excellent", 9, 1)); // 90% → promote
        skills.add_skill(make_skill("good", 7, 3)); // 70% → below 75% threshold
        skills.add_skill(make_skill("poor", 2, 8)); // 20% → no
        skills.add_skill(make_skill("new", 1, 0)); // Too few samples

        let analytics = make_analytics(HashMap::new());
        let policy = RefreshPolicy::default();
        let now = Utc::now();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert_eq!(report.promotions, 1); // Only "excellent"
        let promo = report
            .actions
            .iter()
            .find(|a| a.action_type == RefreshActionType::PromoteSkill)
            .unwrap();
        assert!(promo.description.contains("excellent"));
        assert_eq!(promo.target, "project_brain");
    }

    #[test]
    fn test_flag_undocumented_errors() {
        let mut freqs = HashMap::new();
        freqs.insert("BorrowChecker".to_string(), 15);
        freqs.insert("TypeMismatch".to_string(), 8);
        freqs.insert("Syntax".to_string(), 2); // Below threshold

        let analytics = make_analytics(freqs);
        let policy = RefreshPolicy::default().with_min_error_occurrences(5);
        let now = Utc::now();
        let skills = SkillLibrary::new();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert_eq!(report.undocumented_errors, 2); // BorrowChecker and TypeMismatch
    }

    #[test]
    fn test_documented_errors_not_flagged() {
        let mut freqs = HashMap::new();
        freqs.insert("BorrowChecker".to_string(), 15);
        freqs.insert("TypeMismatch".to_string(), 8);

        let mut documented = HashSet::new();
        documented.insert("BorrowChecker".to_string());

        let analytics = make_analytics(freqs);
        let policy = RefreshPolicy::default()
            .with_min_error_occurrences(5)
            .with_documented_categories(documented);
        let now = Utc::now();
        let skills = SkillLibrary::new();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert_eq!(report.undocumented_errors, 1); // Only TypeMismatch
    }

    #[test]
    fn test_high_frequency_errors_get_high_priority() {
        let mut freqs = HashMap::new();
        freqs.insert("BorrowChecker".to_string(), 20); // 20 >= 5*3 = high
        freqs.insert("TypeMismatch".to_string(), 6); // 6 < 15 = medium

        let analytics = make_analytics(freqs);
        let policy = RefreshPolicy::default().with_min_error_occurrences(5);
        let now = Utc::now();
        let skills = SkillLibrary::new();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        let high = report
            .actions
            .iter()
            .filter(|a| {
                a.action_type == RefreshActionType::FlagUndocumentedError
                    && a.priority == ActionPriority::High
            })
            .count();
        assert_eq!(high, 1); // BorrowChecker
    }

    #[test]
    fn test_combined_analysis() {
        let mut skills = SkillLibrary::new();
        skills.add_skill(make_skill("stale", 1, 0));
        skills.add_skill(make_skill("promote-me", 9, 1));

        let mut freqs = HashMap::new();
        freqs.insert("Async".to_string(), 10);

        let analytics = make_analytics(freqs);
        let policy = RefreshPolicy::default();
        let now = Utc::now();

        let report = analyze_and_refresh(&analytics, &skills, &policy, now);
        assert!(report.has_actions());
        assert_eq!(report.stale_skills, 1);
        assert_eq!(report.promotions, 1);
        assert_eq!(report.undocumented_errors, 1);
        assert_eq!(report.actions.len(), 3);
    }

    #[test]
    fn test_report_display() {
        let report = RefreshReport {
            analyzed_at: "2026-02-22T00:00:00Z".to_string(),
            actions: vec![],
            skills_analyzed: 5,
            error_categories_analyzed: 3,
            stale_skills: 1,
            promotions: 2,
            undocumented_errors: 0,
        };
        let display = report.to_string();
        assert!(display.contains("0 actions"));
        assert!(display.contains("stale=1"));
        assert!(display.contains("promoted=2"));
    }

    #[test]
    fn test_report_serialization() {
        let report = RefreshReport {
            analyzed_at: Utc::now().to_rfc3339(),
            actions: vec![KBRefreshAction {
                action_type: RefreshActionType::PromoteSkill,
                description: "Test".to_string(),
                target: "project_brain".to_string(),
                priority: ActionPriority::Medium,
            }],
            skills_analyzed: 1,
            error_categories_analyzed: 0,
            stale_skills: 0,
            promotions: 1,
            undocumented_errors: 0,
        };

        let json = serde_json::to_string(&report).unwrap();
        let restored: RefreshReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.actions.len(), 1);
        assert_eq!(
            restored.actions[0].action_type,
            RefreshActionType::PromoteSkill
        );
    }

    #[test]
    fn test_policy_builder() {
        let policy = RefreshPolicy::new()
            .with_session_interval(5)
            .with_staleness_days(14)
            .with_min_confidence(0.85)
            .with_min_error_occurrences(3);

        assert_eq!(policy.session_interval, 5);
        assert_eq!(policy.staleness_threshold_days, 14);
        assert_eq!(policy.min_skill_confidence, 0.85);
        assert_eq!(policy.min_error_occurrences, 3);
        assert_eq!(policy.staleness_duration(), Duration::from_secs(14 * 86400));
    }

    #[test]
    fn test_action_type_display() {
        assert_eq!(
            RefreshActionType::DeprecateStalePattern.to_string(),
            "deprecate_stale_pattern"
        );
        assert_eq!(RefreshActionType::PromoteSkill.to_string(), "promote_skill");
        assert_eq!(
            RefreshActionType::FlagUndocumentedError.to_string(),
            "flag_undocumented_error"
        );
    }
}
