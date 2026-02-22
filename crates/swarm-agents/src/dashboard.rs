//! Self-Improvement Metrics Dashboard
//!
//! Provides visibility into the self-improvement loop: skill library health,
//! routing distribution, friction/delight trends, acceptance rates, and
//! escalation patterns. Essential for human oversight of the self-modifying system.
//!
//! # Time Windows
//!
//! All metrics support windowed views: last 24h, 7d, 30d, and all-time.
//! The dashboard computes each window from raw session data, enabling
//! trend detection and anomaly flagging.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use coordination::analytics::skills::SkillLibrary;

use crate::telemetry::{AggregateAnalytics, SessionMetrics};

// ── Time Window ──────────────────────────────────────────────────────

/// Named time window for metric computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeWindow {
    /// Last 24 hours.
    Last24h,
    /// Last 7 days.
    Last7d,
    /// Last 30 days.
    Last30d,
    /// All recorded history.
    AllTime,
}

impl TimeWindow {
    /// Duration of this window. Returns `None` for `AllTime`.
    pub fn duration(&self) -> Option<Duration> {
        match self {
            Self::Last24h => Some(Duration::hours(24)),
            Self::Last7d => Some(Duration::days(7)),
            Self::Last30d => Some(Duration::days(30)),
            Self::AllTime => None,
        }
    }

    /// All standard windows in order.
    pub fn all() -> &'static [TimeWindow] {
        &[
            TimeWindow::Last24h,
            TimeWindow::Last7d,
            TimeWindow::Last30d,
            TimeWindow::AllTime,
        ]
    }
}

impl std::fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Last24h => write!(f, "24h"),
            Self::Last7d => write!(f, "7d"),
            Self::Last30d => write!(f, "30d"),
            Self::AllTime => write!(f, "all-time"),
        }
    }
}

// ── Skill Summary ────────────────────────────────────────────────────

/// Breakdown of skills by confidence bucket.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillSummary {
    /// Total number of skills in the library.
    pub total: usize,
    /// Skills with confidence >= 0.8.
    pub high_confidence: usize,
    /// Skills with confidence >= 0.5 and < 0.8.
    pub medium_confidence: usize,
    /// Skills with confidence < 0.5 (includes untested).
    pub low_confidence: usize,
    /// Average confidence across all skills.
    pub avg_confidence: f64,
}

// ── Windowed Metrics ─────────────────────────────────────────────────

/// Metrics computed for a specific time window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowedMetrics {
    /// Time window these metrics cover.
    pub window: String,
    /// Number of sessions in this window.
    pub session_count: usize,
    /// Success rate (0.0 to 1.0).
    pub success_rate: f64,
    /// Average iterations per session.
    pub avg_iterations: f64,
    /// Escalation rate (sessions that escalated / total sessions).
    pub escalation_rate: f64,
    /// Acceptance rate (sessions that resolved / total sessions).
    pub acceptance_rate: f64,
    /// Routing distribution: model name → count of sessions.
    pub routing_distribution: HashMap<String, usize>,
    /// Average friction score (escalation events + no-change iterations).
    pub friction_score: f64,
    /// Average delight score (first-pass successes + low-iteration sessions).
    pub delight_score: f64,
}

// ── Dashboard Metrics ────────────────────────────────────────────────

/// Complete dashboard output: skill summary + windowed metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    /// When this dashboard was generated.
    pub generated_at: String,
    /// Skill library summary.
    pub skills: SkillSummary,
    /// Metrics per time window.
    pub windows: HashMap<String, WindowedMetrics>,
}

// ── Dashboard Generator ──────────────────────────────────────────────

/// Generate dashboard metrics from session data and skill library.
pub fn generate(
    sessions: &[SessionMetrics],
    skills: &SkillLibrary,
    now: DateTime<Utc>,
) -> DashboardMetrics {
    let skill_summary = summarize_skills(skills);

    let mut windows = HashMap::new();
    for &window in TimeWindow::all() {
        let filtered = filter_sessions(sessions, window, now);
        let metrics = compute_windowed_metrics(&filtered, window);
        windows.insert(window.to_string(), metrics);
    }

    DashboardMetrics {
        generated_at: now.to_rfc3339(),
        skills: skill_summary,
        windows,
    }
}

/// Generate dashboard from an `AggregateAnalytics` (pre-computed) and skills.
///
/// This is a convenience wrapper when you already have aggregate stats
/// but not the raw sessions. It produces an all-time-only view.
pub fn generate_from_aggregate(
    analytics: &AggregateAnalytics,
    skills: &SkillLibrary,
    now: DateTime<Utc>,
) -> DashboardMetrics {
    let skill_summary = summarize_skills(skills);

    let all_time = WindowedMetrics {
        window: TimeWindow::AllTime.to_string(),
        session_count: analytics.total_sessions,
        success_rate: analytics.success_rate,
        avg_iterations: analytics.average_iterations,
        escalation_rate: 0.0, // Not available from aggregate
        acceptance_rate: analytics.success_rate,
        routing_distribution: HashMap::new(), // Not available from aggregate
        friction_score: 0.0,
        delight_score: 0.0,
    };

    let mut windows = HashMap::new();
    windows.insert(TimeWindow::AllTime.to_string(), all_time);

    DashboardMetrics {
        generated_at: now.to_rfc3339(),
        skills: skill_summary,
        windows,
    }
}

// ── Human-Readable Summary ───────────────────────────────────────────

/// Format dashboard metrics as a human-readable CLI summary.
pub fn format_summary(metrics: &DashboardMetrics) -> String {
    let mut lines = Vec::new();

    lines.push("=== Swarm Self-Improvement Dashboard ===".to_string());
    lines.push(format!("Generated: {}", metrics.generated_at));
    lines.push(String::new());

    // Skill library
    lines.push("-- Skill Library --".to_string());
    lines.push(format!(
        "  Total skills: {} (high: {}, medium: {}, low: {})",
        metrics.skills.total,
        metrics.skills.high_confidence,
        metrics.skills.medium_confidence,
        metrics.skills.low_confidence,
    ));
    lines.push(format!(
        "  Avg confidence: {:.1}%",
        metrics.skills.avg_confidence * 100.0
    ));
    lines.push(String::new());

    // Windowed metrics — print in order
    for window_name in &["24h", "7d", "30d", "all-time"] {
        if let Some(w) = metrics.windows.get(*window_name) {
            lines.push(format!("-- {} --", window_name));
            lines.push(format!(
                "  Sessions: {}  Success: {:.0}%  Avg iterations: {:.1}",
                w.session_count,
                w.success_rate * 100.0,
                w.avg_iterations,
            ));
            lines.push(format!(
                "  Escalation: {:.0}%  Acceptance: {:.0}%",
                w.escalation_rate * 100.0,
                w.acceptance_rate * 100.0,
            ));
            lines.push(format!(
                "  Friction: {:.2}  Delight: {:.2}",
                w.friction_score, w.delight_score,
            ));

            if !w.routing_distribution.is_empty() {
                let mut routes: Vec<_> = w.routing_distribution.iter().collect();
                routes.sort_by(|a, b| b.1.cmp(a.1));
                let route_str: Vec<String> =
                    routes.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
                lines.push(format!("  Routing: {}", route_str.join(", ")));
            }
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

// ── Internal Helpers ─────────────────────────────────────────────────

fn summarize_skills(skills: &SkillLibrary) -> SkillSummary {
    let all = skills.skills();
    if all.is_empty() {
        return SkillSummary::default();
    }

    let min_samples = 2;
    let mut high = 0usize;
    let mut medium = 0usize;
    let mut low = 0usize;
    let mut conf_sum = 0.0f64;

    for skill in all {
        let conf = skill.confidence(min_samples);
        conf_sum += conf;
        if conf >= 0.8 {
            high += 1;
        } else if conf >= 0.5 {
            medium += 1;
        } else {
            low += 1;
        }
    }

    SkillSummary {
        total: all.len(),
        high_confidence: high,
        medium_confidence: medium,
        low_confidence: low,
        avg_confidence: conf_sum / all.len() as f64,
    }
}

fn filter_sessions(
    sessions: &[SessionMetrics],
    window: TimeWindow,
    now: DateTime<Utc>,
) -> Vec<&SessionMetrics> {
    let cutoff = window.duration().map(|d| now - d);
    sessions
        .iter()
        .filter(|s| {
            if let Some(cutoff) = cutoff {
                // Parse the timestamp; include session if it's after the cutoff
                if let Ok(ts) = DateTime::parse_from_rfc3339(&s.timestamp) {
                    ts >= cutoff
                } else {
                    // If timestamp can't be parsed, include in all-time only
                    false
                }
            } else {
                // AllTime — include everything
                true
            }
        })
        .collect()
}

fn compute_windowed_metrics(sessions: &[&SessionMetrics], window: TimeWindow) -> WindowedMetrics {
    let count = sessions.len();
    if count == 0 {
        return WindowedMetrics {
            window: window.to_string(),
            ..Default::default()
        };
    }

    let successes = sessions.iter().filter(|s| s.success).count();
    let success_rate = successes as f64 / count as f64;

    let total_iterations: u32 = sessions.iter().map(|s| s.total_iterations).sum();
    let avg_iterations = total_iterations as f64 / count as f64;

    // Escalation rate: sessions where the final tier is not Worker-level
    let escalated = sessions
        .iter()
        .filter(|s| s.iterations.iter().any(|i| i.escalated))
        .count();
    let escalation_rate = escalated as f64 / count as f64;

    // Acceptance rate = success rate (resolved issues / total)
    let acceptance_rate = success_rate;

    // Routing distribution: count coder_route values
    let mut routing = HashMap::new();
    for session in sessions {
        for iter_m in &session.iterations {
            if let Some(route) = &iter_m.coder_route {
                *routing.entry(route.clone()).or_insert(0usize) += 1;
            }
        }
    }

    // Friction score: average (escalations + no-change iterations) per session
    let total_friction: f64 = sessions
        .iter()
        .map(|s| {
            let escalations = s.iterations.iter().filter(|i| i.escalated).count();
            let no_changes = s.total_no_change_iterations as usize;
            (escalations + no_changes) as f64
        })
        .sum();
    let friction_score = total_friction / count as f64;

    // Delight score: fraction of sessions that succeeded on first pass (1 iteration)
    let first_pass = sessions
        .iter()
        .filter(|s| s.success && s.total_iterations == 1)
        .count();
    let delight_score = first_pass as f64 / count as f64;

    WindowedMetrics {
        window: window.to_string(),
        session_count: count,
        success_rate,
        avg_iterations,
        escalation_rate,
        acceptance_rate,
        routing_distribution: routing,
        friction_score,
        delight_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::IterationMetrics;

    fn make_session(
        id: &str,
        success: bool,
        iterations: u32,
        no_change: u32,
        timestamp: &str,
        coder_routes: &[&str],
        has_escalation: bool,
    ) -> SessionMetrics {
        let iter_metrics: Vec<IterationMetrics> = (0..iterations)
            .map(|i| IterationMetrics {
                iteration: i + 1,
                tier: "Worker".to_string(),
                agent_model: "test-model".to_string(),
                agent_prompt_tokens: 100,
                agent_completion_tokens: 200,
                agent_response_ms: 5000,
                verifier_ms: 2000,
                error_count: if success && i + 1 == iterations { 0 } else { 3 },
                error_categories: vec!["TypeMismatch".to_string()],
                no_change: i < no_change,
                auto_fix_applied: false,
                regression_detected: false,
                rollback_performed: false,
                escalated: has_escalation && i == 0,
                coder_route: coder_routes.get(i as usize).map(|s| s.to_string()),
                artifacts: vec![],
                execution_artifact: None,
            })
            .collect();

        SessionMetrics {
            session_id: format!("session-{}", id),
            issue_id: format!("issue-{}", id),
            issue_title: format!("Test issue {}", id),
            success,
            total_iterations: iterations,
            final_tier: if has_escalation {
                "Council".to_string()
            } else {
                "Worker".to_string()
            },
            elapsed_ms: iterations as u64 * 7000,
            total_no_change_iterations: no_change,
            no_change_rate: if iterations > 0 {
                no_change as f64 / iterations as f64
            } else {
                0.0
            },
            cloud_validations: vec![],
            iterations: iter_metrics,
            timestamp: timestamp.to_string(),
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
            approach: "test approach".to_string(),
            success_count: successes,
            failure_count: failures,
        }
    }

    #[test]
    fn test_empty_dashboard() {
        let skills = SkillLibrary::new();
        let now = Utc::now();
        let metrics = generate(&[], &skills, now);

        assert_eq!(metrics.skills.total, 0);
        assert_eq!(metrics.skills.avg_confidence, 0.0);
        assert_eq!(metrics.windows.len(), 4);

        let all_time = &metrics.windows["all-time"];
        assert_eq!(all_time.session_count, 0);
        assert_eq!(all_time.success_rate, 0.0);
    }

    #[test]
    fn test_skill_summary_buckets() {
        let mut skills = SkillLibrary::new().with_min_samples(2);
        skills.add_skill(make_skill("high", 9, 1)); // 0.9 → high
        skills.add_skill(make_skill("medium", 6, 4)); // 0.6 → medium
        skills.add_skill(make_skill("low", 1, 9)); // 0.1 → low
        skills.add_skill(make_skill("untested", 0, 0)); // 0.0 → low (no samples)

        let summary = summarize_skills(&skills);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.high_confidence, 1);
        assert_eq!(summary.medium_confidence, 1);
        assert_eq!(summary.low_confidence, 2);
        assert!(summary.avg_confidence > 0.0);
    }

    #[test]
    fn test_windowed_metrics_computation() {
        let now = Utc::now();
        let recent = (now - Duration::hours(6)).to_rfc3339();
        let last_week = (now - Duration::days(3)).to_rfc3339();
        let old = (now - Duration::days(60)).to_rfc3339();

        let sessions = vec![
            make_session("a", true, 1, 0, &recent, &["RustCoder"], false),
            make_session(
                "b",
                true,
                3,
                1,
                &recent,
                &["RustCoder", "GeneralCoder", "RustCoder"],
                false,
            ),
            make_session("c", false, 5, 2, &last_week, &["GeneralCoder"], true),
            make_session("d", true, 2, 0, &old, &["RustCoder", "RustCoder"], false),
        ];

        let skills = SkillLibrary::new();
        let metrics = generate(&sessions, &skills, now);

        // 24h window: sessions a, b
        let w24h = &metrics.windows["24h"];
        assert_eq!(w24h.session_count, 2);
        assert_eq!(w24h.success_rate, 1.0); // both successful
        assert!((w24h.avg_iterations - 2.0).abs() < 0.01); // (1+3)/2
        assert_eq!(w24h.escalation_rate, 0.0);
        assert!(w24h.delight_score > 0.0); // session a is first-pass

        // 7d window: sessions a, b, c
        let w7d = &metrics.windows["7d"];
        assert_eq!(w7d.session_count, 3);
        assert!((w7d.success_rate - 2.0 / 3.0).abs() < 0.01);

        // all-time: all 4 sessions
        let wall = &metrics.windows["all-time"];
        assert_eq!(wall.session_count, 4);
        assert!((wall.success_rate - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_routing_distribution() {
        let now = Utc::now();
        let ts = now.to_rfc3339();

        let sessions = vec![
            make_session("a", true, 2, 0, &ts, &["RustCoder", "RustCoder"], false),
            make_session("b", true, 1, 0, &ts, &["GeneralCoder"], false),
        ];

        let skills = SkillLibrary::new();
        let metrics = generate(&sessions, &skills, now);
        let wall = &metrics.windows["all-time"];

        assert_eq!(wall.routing_distribution.get("RustCoder"), Some(&2));
        assert_eq!(wall.routing_distribution.get("GeneralCoder"), Some(&1));
    }

    #[test]
    fn test_friction_and_delight_scores() {
        let now = Utc::now();
        let ts = now.to_rfc3339();

        let sessions = vec![
            // First-pass success → high delight, zero friction
            make_session("a", true, 1, 0, &ts, &[], false),
            // Escalation + no-change → friction
            make_session("b", false, 4, 2, &ts, &[], true),
        ];

        let skills = SkillLibrary::new();
        let metrics = generate(&sessions, &skills, now);
        let wall = &metrics.windows["all-time"];

        // Delight: 1 first-pass out of 2 sessions = 0.5
        assert!((wall.delight_score - 0.5).abs() < 0.01);

        // Friction: session a = 0, session b = 1 escalation + 2 no-change = 3
        // Average = (0 + 3) / 2 = 1.5
        assert!((wall.friction_score - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_escalation_rate() {
        let now = Utc::now();
        let ts = now.to_rfc3339();

        let sessions = vec![
            make_session("a", true, 1, 0, &ts, &[], false),
            make_session("b", false, 3, 0, &ts, &[], true),
            make_session("c", true, 2, 0, &ts, &[], true),
        ];

        let skills = SkillLibrary::new();
        let metrics = generate(&sessions, &skills, now);
        let wall = &metrics.windows["all-time"];

        // 2 out of 3 sessions had escalation
        assert!((wall.escalation_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_format_summary_output() {
        let now = Utc::now();
        let ts = now.to_rfc3339();

        let sessions = vec![
            make_session("a", true, 1, 0, &ts, &["RustCoder"], false),
            make_session("b", true, 3, 1, &ts, &["GeneralCoder"], false),
        ];

        let mut skills = SkillLibrary::new();
        skills.add_skill(make_skill("borrow-fix", 8, 2));

        let metrics = generate(&sessions, &skills, now);
        let summary = format_summary(&metrics);

        assert!(summary.contains("Self-Improvement Dashboard"));
        assert!(summary.contains("Skill Library"));
        assert!(summary.contains("Total skills: 1"));
        assert!(summary.contains("all-time"));
        assert!(summary.contains("Sessions: 2"));
    }

    #[test]
    fn test_generate_from_aggregate() {
        let analytics = AggregateAnalytics {
            total_sessions: 10,
            success_rate: 0.8,
            average_iterations: 2.5,
            average_elapsed_ms: 15000.0,
            total_prompt_tokens: 50000,
            total_completion_tokens: 30000,
            error_category_frequencies: HashMap::new(),
        };

        let skills = SkillLibrary::new();
        let now = Utc::now();
        let metrics = generate_from_aggregate(&analytics, &skills, now);

        assert_eq!(metrics.windows.len(), 1);
        let wall = &metrics.windows["all-time"];
        assert_eq!(wall.session_count, 10);
        assert_eq!(wall.success_rate, 0.8);
        assert_eq!(wall.avg_iterations, 2.5);
    }

    #[test]
    fn test_dashboard_serialization() {
        let now = Utc::now();
        let skills = SkillLibrary::new();
        let metrics = generate(&[], &skills, now);

        let json = serde_json::to_string(&metrics).unwrap();
        let restored: DashboardMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.skills.total, 0);
        assert_eq!(restored.windows.len(), 4);
    }

    #[test]
    fn test_time_window_display() {
        assert_eq!(TimeWindow::Last24h.to_string(), "24h");
        assert_eq!(TimeWindow::Last7d.to_string(), "7d");
        assert_eq!(TimeWindow::Last30d.to_string(), "30d");
        assert_eq!(TimeWindow::AllTime.to_string(), "all-time");
    }

    #[test]
    fn test_time_window_duration() {
        assert!(TimeWindow::Last24h.duration().is_some());
        assert!(TimeWindow::AllTime.duration().is_none());
        assert_eq!(TimeWindow::Last7d.duration().unwrap().num_days(), 7);
    }
}
