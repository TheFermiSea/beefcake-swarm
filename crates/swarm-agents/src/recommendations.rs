//! Experiment-ready variant recommendations from Autopilot analysis.

use serde::{Deserialize, Serialize};

use crate::meta_reflection::{InsightType, MetaInsight};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentPriority {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCandidate {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecommendation {
    pub experiment_id: String,
    pub hypothesis: String,
    pub variants: Vec<VariantCandidate>,
    pub priority: ExperimentPriority,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

pub fn generate_recommendations(insights: &[MetaInsight]) -> Vec<ExperimentRecommendation> {
    insights
        .iter()
        .filter(|i| i.confidence >= 0.5)
        .map(insight_to_recommendation)
        .collect()
}

fn insight_to_recommendation(insight: &MetaInsight) -> ExperimentRecommendation {
    let (category, variants) = match insight.insight_type {
        InsightType::ModelPerformance => (
            "model_perf",
            vec![
                VariantCandidate {
                    name: "control".into(),
                    description: "Current model.".into(),
                },
                VariantCandidate {
                    name: "challenger".into(),
                    description: format!("Apply: {}", insight.recommendation),
                },
            ],
        ),
        InsightType::ErrorPattern => (
            "error_pattern",
            vec![
                VariantCandidate {
                    name: "control".into(),
                    description: "Existing handling.".into(),
                },
                VariantCandidate {
                    name: "targeted".into(),
                    description: format!("Apply: {}", insight.recommendation),
                },
            ],
        ),
        InsightType::RoutingAdjustment => (
            "routing",
            vec![
                VariantCandidate {
                    name: "current".into(),
                    description: "Existing routing.".into(),
                },
                VariantCandidate {
                    name: "adjusted".into(),
                    description: format!("Apply: {}", insight.recommendation),
                },
            ],
        ),
        InsightType::PromptPerformance => (
            "prompt_perf",
            vec![
                VariantCandidate {
                    name: "prompt_control".into(),
                    description: "Current prompt.".into(),
                },
                VariantCandidate {
                    name: "prompt_challenger".into(),
                    description: format!("Apply: {}", insight.recommendation),
                },
            ],
        ),
        InsightType::SkillPromotion => (
            "skill_promo",
            vec![
                VariantCandidate {
                    name: "optional".into(),
                    description: "Skill on request only.".into(),
                },
                VariantCandidate {
                    name: "default".into(),
                    description: format!("Promote: {}", insight.recommendation),
                },
            ],
        ),
    };
    ExperimentRecommendation {
        experiment_id: make_id(category, &insight.description),
        hypothesis: insight.description.clone(),
        variants,
        priority: priority_from_confidence(insight.confidence),
        confidence: insight.confidence,
        evidence: insight.evidence.clone(),
    }
}

fn make_id(category: &str, desc: &str) -> String {
    let slug: String = desc
        .chars()
        .take(32)
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("{category}_{slug}")
}

fn priority_from_confidence(confidence: f64) -> ExperimentPriority {
    if confidence >= 0.85 {
        ExperimentPriority::High
    } else if confidence >= 0.65 {
        ExperimentPriority::Medium
    } else {
        ExperimentPriority::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_insight(t: InsightType, confidence: f64) -> MetaInsight {
        MetaInsight {
            timestamp: chrono::Utc::now(),
            insight_type: t,
            description: "Test insight".into(),
            recommendation: "Apply change".into(),
            confidence,
            evidence: vec!["issue-42".into()],
        }
    }

    #[test]
    fn filters_low_confidence() {
        let insights = vec![make_insight(InsightType::ModelPerformance, 0.3)];
        assert!(generate_recommendations(&insights).is_empty());
    }

    #[test]
    fn passes_sufficient_confidence() {
        let insights = vec![make_insight(InsightType::ModelPerformance, 0.5)];
        assert_eq!(generate_recommendations(&insights).len(), 1);
    }

    #[test]
    fn each_type_has_two_variants() {
        for t in [
            InsightType::ModelPerformance,
            InsightType::ErrorPattern,
            InsightType::RoutingAdjustment,
            InsightType::PromptPerformance,
            InsightType::SkillPromotion,
        ] {
            let recs = generate_recommendations(&[make_insight(t, 0.8)]);
            assert_eq!(recs[0].variants.len(), 2);
        }
    }

    #[test]
    fn empty_input() {
        assert!(generate_recommendations(&[]).is_empty());
    }

    #[test]
    fn evidence_propagated() {
        let recs = generate_recommendations(&[make_insight(InsightType::SkillPromotion, 0.8)]);
        assert_eq!(recs[0].evidence, vec!["issue-42"]);
    }
}
