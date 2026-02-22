//! SLO Definitions and Dashboard Spec
//!
//! Defines service-level objectives for swarm quality metrics and provides
//! evaluation against the [`OrchestrationMetrics`] computed by the benchmark
//! harness. Each SLO has warning and critical thresholds with alert support.
//!
//! # SLO Targets
//!
//! | Metric | Warning | Critical | Direction |
//! |---|---|---|---|
//! | Verifier first-pass rate | ≥ 0.30 | ≥ 0.20 | Higher is better |
//! | Overall success rate | ≥ 0.70 | ≥ 0.50 | Higher is better |
//! | Escalation ratio | ≤ 0.40 | ≤ 0.60 | Lower is better |
//! | P95 latency | ≤ 600s | ≤ 900s | Lower is better |
//! | Cost per closed issue | ≤ $0.50 | ≤ $1.00 | Lower is better |
//! | Stuck rate | ≤ 0.15 | ≤ 0.25 | Lower is better |
//!
//! # Dashboard Panels
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ Swarm Quality Dashboard                                 │
//! ├─────────────┬─────────────┬─────────────┬──────────────┤
//! │ First-Pass  │ Success     │ Escalation  │ Stuck        │
//! │ Rate        │ Rate        │ Rate        │ Rate         │
//! │ [gauge]     │ [gauge]     │ [gauge]     │ [gauge]      │
//! ├─────────────┴─────────────┴─────────────┴──────────────┤
//! │ Latency Distribution (p50 / p95 / max)                  │
//! │ [histogram]                                             │
//! ├─────────────────────────────────────────────────────────┤
//! │ Cost per Issue (rolling avg)                            │
//! │ [time series]                                           │
//! ├─────────────────────────────────────────────────────────┤
//! │ SLO Compliance Timeline                                 │
//! │ [status timeline: green/yellow/red per SLO]             │
//! └─────────────────────────────────────────────────────────┘
//! ```

use crate::benchmark::OrchestrationMetrics;
use serde::{Deserialize, Serialize};

/// Alert severity level for SLO violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// All clear — metric within target.
    Ok,
    /// Approaching SLO boundary.
    Warning,
    /// SLO violated.
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Direction of a metric (whether higher or lower is better).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricDirection {
    /// Higher values are better (e.g., success rate).
    HigherIsBetter,
    /// Lower values are better (e.g., latency, cost).
    LowerIsBetter,
}

/// An individual SLO target with warning and critical thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloTarget {
    /// Human-readable name (e.g., "First-pass verifier rate").
    pub name: String,
    /// Which `OrchestrationMetrics` field this maps to.
    pub metric_field: MetricField,
    /// Warning threshold value.
    pub warning_threshold: f64,
    /// Critical threshold value.
    pub critical_threshold: f64,
    /// Whether higher or lower is better.
    pub direction: MetricDirection,
    /// Unit label for display (e.g., "%", "s", "$").
    pub unit: String,
}

impl SloTarget {
    /// Evaluate a metric value against this SLO's thresholds.
    pub fn evaluate(&self, value: f64) -> AlertSeverity {
        match self.direction {
            MetricDirection::HigherIsBetter => {
                if value >= self.warning_threshold {
                    AlertSeverity::Ok
                } else if value >= self.critical_threshold {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Critical
                }
            }
            MetricDirection::LowerIsBetter => {
                if value <= self.warning_threshold {
                    AlertSeverity::Ok
                } else if value <= self.critical_threshold {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Critical
                }
            }
        }
    }
}

/// Which field of `OrchestrationMetrics` an SLO maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricField {
    FirstPassRate,
    OverallSuccessRate,
    EscalationRate,
    LatencyP95,
    CostPerIssue,
    StuckRate,
}

impl std::fmt::Display for MetricField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstPassRate => write!(f, "first_pass_rate"),
            Self::OverallSuccessRate => write!(f, "overall_success_rate"),
            Self::EscalationRate => write!(f, "escalation_rate"),
            Self::LatencyP95 => write!(f, "latency_p95"),
            Self::CostPerIssue => write!(f, "cost_per_issue"),
            Self::StuckRate => write!(f, "stuck_rate"),
        }
    }
}

/// Result of evaluating a single SLO against live metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloResult {
    /// The SLO target being evaluated.
    pub target: SloTarget,
    /// The observed metric value.
    pub observed: f64,
    /// Alert severity from the evaluation.
    pub severity: AlertSeverity,
}

impl SloResult {
    /// Whether this SLO is violated (warning or critical).
    pub fn is_violated(&self) -> bool {
        self.severity != AlertSeverity::Ok
    }
}

/// Aggregate SLO evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloReport {
    /// Individual SLO results.
    pub results: Vec<SloResult>,
    /// Overall worst severity across all SLOs.
    pub overall_severity: AlertSeverity,
    /// Number of SLOs passing.
    pub passing: usize,
    /// Number of SLOs at warning.
    pub warnings: usize,
    /// Number of SLOs at critical.
    pub critical: usize,
}

impl SloReport {
    /// Whether all SLOs are passing.
    pub fn all_passing(&self) -> bool {
        self.overall_severity == AlertSeverity::Ok
    }

    /// Format as a human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "SLO Report: {} passing, {} warning, {} critical [{}]",
            self.passing, self.warnings, self.critical, self.overall_severity,
        ));
        lines.push(String::new());

        for r in &self.results {
            let status = match r.severity {
                AlertSeverity::Ok => "[PASS]",
                AlertSeverity::Warning => "[WARN]",
                AlertSeverity::Critical => "[CRIT]",
            };
            let threshold_info = match r.target.direction {
                MetricDirection::HigherIsBetter => format!(
                    "warn >= {:.2}{} / crit >= {:.2}{}",
                    r.target.warning_threshold,
                    r.target.unit,
                    r.target.critical_threshold,
                    r.target.unit,
                ),
                MetricDirection::LowerIsBetter => format!(
                    "warn <= {:.2}{} / crit <= {:.2}{}",
                    r.target.warning_threshold,
                    r.target.unit,
                    r.target.critical_threshold,
                    r.target.unit,
                ),
            };
            lines.push(format!(
                "  {} {}: {:.2}{} ({})",
                status, r.target.name, r.observed, r.target.unit, threshold_info,
            ));
        }

        lines.join("\n")
    }
}

/// The default SLO target set for the swarm.
///
/// These values are intentionally conservative for initial rollout and
/// can be tightened as the system matures.
pub fn default_slo_targets() -> Vec<SloTarget> {
    vec![
        SloTarget {
            name: "First-pass verifier rate".to_string(),
            metric_field: MetricField::FirstPassRate,
            warning_threshold: 0.30,
            critical_threshold: 0.20,
            direction: MetricDirection::HigherIsBetter,
            unit: "".to_string(),
        },
        SloTarget {
            name: "Overall success rate".to_string(),
            metric_field: MetricField::OverallSuccessRate,
            warning_threshold: 0.70,
            critical_threshold: 0.50,
            direction: MetricDirection::HigherIsBetter,
            unit: "".to_string(),
        },
        SloTarget {
            name: "Escalation ratio".to_string(),
            metric_field: MetricField::EscalationRate,
            warning_threshold: 0.40,
            critical_threshold: 0.60,
            direction: MetricDirection::LowerIsBetter,
            unit: "".to_string(),
        },
        SloTarget {
            name: "P95 latency".to_string(),
            metric_field: MetricField::LatencyP95,
            warning_threshold: 600.0,
            critical_threshold: 900.0,
            direction: MetricDirection::LowerIsBetter,
            unit: "s".to_string(),
        },
        SloTarget {
            name: "Cost per closed issue".to_string(),
            metric_field: MetricField::CostPerIssue,
            warning_threshold: 0.50,
            critical_threshold: 1.00,
            direction: MetricDirection::LowerIsBetter,
            unit: "$".to_string(),
        },
        SloTarget {
            name: "Stuck rate".to_string(),
            metric_field: MetricField::StuckRate,
            warning_threshold: 0.15,
            critical_threshold: 0.25,
            direction: MetricDirection::LowerIsBetter,
            unit: "".to_string(),
        },
    ]
}

/// Extract the relevant metric value from `OrchestrationMetrics` for a given field.
pub fn extract_metric(metrics: &OrchestrationMetrics, field: MetricField) -> f64 {
    match field {
        MetricField::FirstPassRate => metrics.first_pass_rate,
        MetricField::OverallSuccessRate => metrics.overall_success_rate,
        MetricField::EscalationRate => metrics.escalation_rate,
        MetricField::LatencyP95 => metrics.latency_p95.as_secs_f64(),
        MetricField::CostPerIssue => metrics.cost_avg,
        MetricField::StuckRate => metrics.stuck_rate,
    }
}

/// Evaluate all default SLOs against computed orchestration metrics.
pub fn evaluate_slos(metrics: &OrchestrationMetrics) -> SloReport {
    evaluate_slos_with_targets(metrics, &default_slo_targets())
}

/// Evaluate a custom set of SLO targets against computed orchestration metrics.
pub fn evaluate_slos_with_targets(
    metrics: &OrchestrationMetrics,
    targets: &[SloTarget],
) -> SloReport {
    let mut results = Vec::with_capacity(targets.len());
    let mut overall = AlertSeverity::Ok;
    let mut passing = 0;
    let mut warnings = 0;
    let mut critical = 0;

    for target in targets {
        let observed = extract_metric(metrics, target.metric_field);
        let severity = target.evaluate(observed);

        if severity > overall {
            overall = severity;
        }
        match severity {
            AlertSeverity::Ok => passing += 1,
            AlertSeverity::Warning => warnings += 1,
            AlertSeverity::Critical => critical += 1,
        }

        results.push(SloResult {
            target: target.clone(),
            observed,
            severity,
        });
    }

    SloReport {
        results,
        overall_severity: overall,
        passing,
        warnings,
        critical,
    }
}

/// Dashboard panel specification for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    /// Panel title.
    pub title: String,
    /// Panel type for rendering.
    pub panel_type: PanelType,
    /// Metrics displayed in this panel.
    pub metric_fields: Vec<MetricField>,
    /// Display width (1-4 columns).
    pub width: u8,
}

/// Panel visualization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelType {
    /// Single-value gauge with SLO color coding.
    Gauge,
    /// Distribution histogram (p50/p95/max).
    Histogram,
    /// Time series line chart.
    TimeSeries,
    /// Status timeline (green/yellow/red bands).
    StatusTimeline,
}

/// Dashboard layout specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSpec {
    /// Dashboard title.
    pub title: String,
    /// Ordered list of panels.
    pub panels: Vec<DashboardPanel>,
    /// Refresh interval in seconds.
    pub refresh_interval_secs: u64,
}

/// Default dashboard specification for the swarm quality dashboard.
pub fn default_dashboard_spec() -> DashboardSpec {
    DashboardSpec {
        title: "Swarm Quality Dashboard".to_string(),
        refresh_interval_secs: 60,
        panels: vec![
            DashboardPanel {
                title: "First-Pass Rate".to_string(),
                panel_type: PanelType::Gauge,
                metric_fields: vec![MetricField::FirstPassRate],
                width: 1,
            },
            DashboardPanel {
                title: "Success Rate".to_string(),
                panel_type: PanelType::Gauge,
                metric_fields: vec![MetricField::OverallSuccessRate],
                width: 1,
            },
            DashboardPanel {
                title: "Escalation Rate".to_string(),
                panel_type: PanelType::Gauge,
                metric_fields: vec![MetricField::EscalationRate],
                width: 1,
            },
            DashboardPanel {
                title: "Stuck Rate".to_string(),
                panel_type: PanelType::Gauge,
                metric_fields: vec![MetricField::StuckRate],
                width: 1,
            },
            DashboardPanel {
                title: "Latency Distribution".to_string(),
                panel_type: PanelType::Histogram,
                metric_fields: vec![MetricField::LatencyP95],
                width: 4,
            },
            DashboardPanel {
                title: "Cost per Issue".to_string(),
                panel_type: PanelType::TimeSeries,
                metric_fields: vec![MetricField::CostPerIssue],
                width: 4,
            },
            DashboardPanel {
                title: "SLO Compliance".to_string(),
                panel_type: PanelType::StatusTimeline,
                metric_fields: vec![
                    MetricField::FirstPassRate,
                    MetricField::OverallSuccessRate,
                    MetricField::EscalationRate,
                    MetricField::LatencyP95,
                    MetricField::CostPerIssue,
                    MetricField::StuckRate,
                ],
                width: 4,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Helper to build metrics with specific values.
    fn make_metrics(
        first_pass_rate: f64,
        success_rate: f64,
        escalation_rate: f64,
        latency_p95_secs: u64,
        cost_avg: f64,
        stuck_rate: f64,
    ) -> OrchestrationMetrics {
        OrchestrationMetrics {
            session_count: 100,
            first_pass_rate,
            overall_success_rate: success_rate,
            avg_iterations_to_green: 3.0,
            median_iterations_to_green: 2.0,
            escalation_rate,
            avg_escalations: 1.0,
            latency_p50: Duration::from_secs(120),
            latency_p95: Duration::from_secs(latency_p95_secs),
            latency_max: Duration::from_secs(1200),
            tokens_p50: 10_000,
            tokens_p95: 50_000,
            tokens_total: 1_000_000,
            cost_total: cost_avg * 100.0,
            cost_avg,
            stuck_rate,
        }
    }

    #[test]
    fn test_all_slos_passing() {
        let metrics = make_metrics(0.45, 0.85, 0.25, 400, 0.30, 0.05);
        let report = evaluate_slos(&metrics);

        assert!(report.all_passing());
        assert_eq!(report.overall_severity, AlertSeverity::Ok);
        assert_eq!(report.passing, 6);
        assert_eq!(report.warnings, 0);
        assert_eq!(report.critical, 0);
    }

    #[test]
    fn test_warning_threshold() {
        // First-pass rate just below warning (0.30), above critical (0.20)
        let metrics = make_metrics(0.25, 0.85, 0.25, 400, 0.30, 0.05);
        let report = evaluate_slos(&metrics);

        assert!(!report.all_passing());
        assert_eq!(report.overall_severity, AlertSeverity::Warning);
        assert_eq!(report.warnings, 1);
        assert_eq!(report.passing, 5);
    }

    #[test]
    fn test_critical_threshold() {
        // First-pass rate below critical (0.20)
        let metrics = make_metrics(0.10, 0.85, 0.25, 400, 0.30, 0.05);
        let report = evaluate_slos(&metrics);

        assert!(!report.all_passing());
        assert_eq!(report.overall_severity, AlertSeverity::Critical);
        assert_eq!(report.critical, 1);
    }

    #[test]
    fn test_latency_slo() {
        // P95 latency above warning (600s) but below critical (900s)
        let metrics = make_metrics(0.45, 0.85, 0.25, 750, 0.30, 0.05);
        let report = evaluate_slos(&metrics);

        let latency_result = report
            .results
            .iter()
            .find(|r| r.target.metric_field == MetricField::LatencyP95)
            .unwrap();
        assert_eq!(latency_result.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_cost_slo() {
        // Cost above critical ($1.00)
        let metrics = make_metrics(0.45, 0.85, 0.25, 400, 1.50, 0.05);
        let report = evaluate_slos(&metrics);

        let cost_result = report
            .results
            .iter()
            .find(|r| r.target.metric_field == MetricField::CostPerIssue)
            .unwrap();
        assert_eq!(cost_result.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_stuck_rate_slo() {
        // Stuck rate above warning (0.15) but below critical (0.25)
        let metrics = make_metrics(0.45, 0.85, 0.25, 400, 0.30, 0.20);
        let report = evaluate_slos(&metrics);

        let stuck_result = report
            .results
            .iter()
            .find(|r| r.target.metric_field == MetricField::StuckRate)
            .unwrap();
        assert_eq!(stuck_result.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_multiple_violations() {
        // Several metrics in violation at once
        let metrics = make_metrics(0.10, 0.40, 0.70, 1000, 2.00, 0.30);
        let report = evaluate_slos(&metrics);

        assert_eq!(report.overall_severity, AlertSeverity::Critical);
        // All 6 should be critical
        assert_eq!(report.critical, 6);
        assert_eq!(report.passing, 0);
    }

    #[test]
    fn test_exact_threshold_values() {
        // Values exactly at warning thresholds
        let metrics = make_metrics(0.30, 0.70, 0.40, 600, 0.50, 0.15);
        let report = evaluate_slos(&metrics);

        // At threshold means passing (>= for higher-is-better, <= for lower-is-better)
        assert!(report.all_passing());
    }

    #[test]
    fn test_slo_result_is_violated() {
        let target = SloTarget {
            name: "test".to_string(),
            metric_field: MetricField::FirstPassRate,
            warning_threshold: 0.50,
            critical_threshold: 0.30,
            direction: MetricDirection::HigherIsBetter,
            unit: "".to_string(),
        };

        let ok_result = SloResult {
            target: target.clone(),
            observed: 0.60,
            severity: AlertSeverity::Ok,
        };
        assert!(!ok_result.is_violated());

        let warn_result = SloResult {
            target,
            observed: 0.40,
            severity: AlertSeverity::Warning,
        };
        assert!(warn_result.is_violated());
    }

    #[test]
    fn test_report_summary_format() {
        let metrics = make_metrics(0.45, 0.85, 0.25, 750, 0.30, 0.05);
        let report = evaluate_slos(&metrics);
        let summary = report.summary();

        assert!(summary.contains("SLO Report:"));
        assert!(summary.contains("[PASS]"));
        assert!(summary.contains("[WARN]")); // latency at 750s
        assert!(summary.contains("P95 latency"));
    }

    #[test]
    fn test_default_dashboard_spec() {
        let spec = default_dashboard_spec();
        assert_eq!(spec.title, "Swarm Quality Dashboard");
        assert_eq!(spec.panels.len(), 7);
        assert_eq!(spec.refresh_interval_secs, 60);

        // First 4 panels are gauges
        for panel in &spec.panels[..4] {
            assert_eq!(panel.panel_type, PanelType::Gauge);
            assert_eq!(panel.width, 1);
        }

        // Last panel is compliance timeline
        let compliance = &spec.panels[6];
        assert_eq!(compliance.panel_type, PanelType::StatusTimeline);
        assert_eq!(compliance.metric_fields.len(), 6);
    }

    #[test]
    fn test_custom_slo_targets() {
        let targets = vec![SloTarget {
            name: "Custom rate".to_string(),
            metric_field: MetricField::FirstPassRate,
            warning_threshold: 0.90,
            critical_threshold: 0.80,
            direction: MetricDirection::HigherIsBetter,
            unit: "".to_string(),
        }];

        let metrics = make_metrics(0.85, 0.85, 0.25, 400, 0.30, 0.05);
        let report = evaluate_slos_with_targets(&metrics, &targets);

        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Ok < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    #[test]
    fn test_metric_field_display() {
        assert_eq!(MetricField::FirstPassRate.to_string(), "first_pass_rate");
        assert_eq!(MetricField::LatencyP95.to_string(), "latency_p95");
        assert_eq!(MetricField::CostPerIssue.to_string(), "cost_per_issue");
    }

    #[test]
    fn test_slo_report_json_roundtrip() {
        let metrics = make_metrics(0.45, 0.85, 0.25, 400, 0.30, 0.05);
        let report = evaluate_slos(&metrics);

        let json = serde_json::to_string(&report).unwrap();
        let restored: SloReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.passing, report.passing);
        assert_eq!(restored.overall_severity, report.overall_severity);
        assert_eq!(restored.results.len(), report.results.len());
    }

    #[test]
    fn test_dashboard_spec_json_roundtrip() {
        let spec = default_dashboard_spec();
        let json = serde_json::to_string(&spec).unwrap();
        let restored: DashboardSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.panels.len(), spec.panels.len());
        assert_eq!(restored.title, spec.title);
    }
}
