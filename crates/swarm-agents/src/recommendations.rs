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
    /// Optional model identifier this variant should use (e.g. "claude-opus-4-6").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_hint: Option<String>,
    /// Optional prompt-version tag this variant should use (e.g. "v2-concise").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecommendation {
    pub experiment_id: String,
    pub hypothesis: String,
    pub variants: Vec<VariantCandidate>,
    pub priority: ExperimentPriority,
    pub confidence: f64,
    pub evidence: Vec<String>,
    /// The TensorZero function name (or logical target) this experiment addresses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_function: Option<String>,
    /// Observed failure modes that motivated this recommendation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failure_modes: Vec<String>,
}

/// Flat, machine-readable record consumed by an experiment runner or human review.
///
/// Each `ExperimentRecommendation` expands into one `RunnerInput` per variant so
/// the runner can dispatch each variant independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerInput {
    pub experiment_id: String,
    pub variant_name: String,
    pub hypothesis: String,
    pub priority: ExperimentPriority,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_function: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_hint: Option<String>,
    pub failure_modes: Vec<String>,
    pub evidence: Vec<String>,
}

/// Expand a slice of recommendations into flat `RunnerInput` records.
pub fn to_runner_inputs(recs: &[ExperimentRecommendation]) -> Vec<RunnerInput> {
    recs.iter()
        .flat_map(|rec| {
            rec.variants.iter().map(move |v| RunnerInput {
                experiment_id: rec.experiment_id.clone(),
                variant_name: v.name.clone(),
                hypothesis: rec.hypothesis.clone(),
                priority: rec.priority.clone(),
                confidence: rec.confidence,
                target_function: rec.target_function.clone(),
                model_hint: v.model_hint.clone(),
                prompt_hint: v.prompt_hint.clone(),
                failure_modes: rec.failure_modes.clone(),
                evidence: rec.evidence.clone(),
            })
        })
        .collect()
}

pub fn generate_recommendations(insights: &[MetaInsight]) -> Vec<ExperimentRecommendation> {
    insights
        .iter()
        .filter(|i| i.confidence >= 0.5)
        .map(insight_to_recommendation)
        .collect()
}

fn insight_to_recommendation(insight: &MetaInsight) -> ExperimentRecommendation {
    let (category, target_function, variants) = match insight.insight_type {
        InsightType::ModelPerformance => (
            "model_perf",
            Some("swarm_model_dispatch".to_string()),
            vec![
                VariantCandidate {
                    name: "control".into(),
                    description: "Current model.".into(),
                    model_hint: None,
                    prompt_hint: None,
                },
                VariantCandidate {
                    name: "challenger".into(),
                    description: format!("Apply: {}", insight.recommendation),
                    model_hint: Some("challenger_model".into()),
                    prompt_hint: None,
                },
            ],
        ),
        InsightType::ErrorPattern => (
            "error_pattern",
            Some("swarm_error_handler".to_string()),
            vec![
                VariantCandidate {
                    name: "control".into(),
                    description: "Existing handling.".into(),
                    model_hint: None,
                    prompt_hint: None,
                },
                VariantCandidate {
                    name: "targeted".into(),
                    description: format!("Apply: {}", insight.recommendation),
                    model_hint: None,
                    prompt_hint: Some("error_targeted_v1".into()),
                },
            ],
        ),
        InsightType::RoutingAdjustment => (
            "routing",
            Some("swarm_router".to_string()),
            vec![
                VariantCandidate {
                    name: "current".into(),
                    description: "Existing routing.".into(),
                    model_hint: None,
                    prompt_hint: None,
                },
                VariantCandidate {
                    name: "adjusted".into(),
                    description: format!("Apply: {}", insight.recommendation),
                    model_hint: None,
                    prompt_hint: Some("routing_adjusted_v1".into()),
                },
            ],
        ),
        InsightType::PromptPerformance => (
            "prompt_perf",
            Some("swarm_prompt_engine".to_string()),
            vec![
                VariantCandidate {
                    name: "prompt_control".into(),
                    description: "Current prompt.".into(),
                    model_hint: None,
                    prompt_hint: Some("current".into()),
                },
                VariantCandidate {
                    name: "prompt_challenger".into(),
                    description: format!("Apply: {}", insight.recommendation),
                    model_hint: None,
                    prompt_hint: Some("challenger_v1".into()),
                },
            ],
        ),
        InsightType::SkillPromotion => (
            "skill_promo",
            Some("swarm_skill_selector".to_string()),
            vec![
                VariantCandidate {
                    name: "optional".into(),
                    description: "Skill on request only.".into(),
                    model_hint: None,
                    prompt_hint: None,
                },
                VariantCandidate {
                    name: "default".into(),
                    description: format!("Promote: {}", insight.recommendation),
                    model_hint: None,
                    prompt_hint: Some("skill_promoted_v1".into()),
                },
            ],
        ),
    };
    // Derive failure modes from the evidence list (each evidence entry is an issue ID
    // or a human-readable failure description surfaced by MetaReflector).
    let failure_modes: Vec<String> = insight
        .evidence
        .iter()
        .filter(|e| !e.starts_with("issue-") && !e.is_empty())
        .cloned()
        .chain(std::iter::once(insight.description.clone()))
        .collect();
    ExperimentRecommendation {
        experiment_id: make_id(category, &insight.description),
        hypothesis: insight.description.clone(),
        variants,
        priority: priority_from_confidence(insight.confidence),
        confidence: insight.confidence,
        evidence: insight.evidence.clone(),
        target_function,
        failure_modes,
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
    fn target_function_is_set() {
        for t in [
            InsightType::ModelPerformance,
            InsightType::ErrorPattern,
            InsightType::RoutingAdjustment,
            InsightType::PromptPerformance,
            InsightType::SkillPromotion,
        ] {
            let recs = generate_recommendations(&[make_insight(t.clone(), 0.8)]);
            assert!(
                recs[0].target_function.is_some(),
                "target_function must be set for {:?}",
                t
            );
        }
    }

    #[test]
    fn failure_modes_include_description() {
        let recs = generate_recommendations(&[make_insight(InsightType::ModelPerformance, 0.8)]);
        assert!(recs[0].failure_modes.contains(&"Test insight".to_string()));
    }

    #[test]
    fn runner_inputs_expand_variants() {
        use super::to_runner_inputs;
        let recs = generate_recommendations(&[make_insight(InsightType::PromptPerformance, 0.8)]);
        let inputs = to_runner_inputs(&recs);
        // PromptPerformance produces 2 variants → 2 RunnerInput rows
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].experiment_id, inputs[1].experiment_id);
        // challenger variant should carry a prompt_hint
        let challenger = inputs
            .iter()
            .find(|i| i.variant_name == "prompt_challenger")
            .unwrap();
        assert!(challenger.prompt_hint.is_some());
    }

    #[test]
    fn model_hint_set_for_model_performance() {
        use super::to_runner_inputs;
        let recs = generate_recommendations(&[make_insight(InsightType::ModelPerformance, 0.8)]);
        let inputs = to_runner_inputs(&recs);
        let challenger = inputs
            .iter()
            .find(|i| i.variant_name == "challenger")
            .unwrap();
        assert!(challenger.model_hint.is_some());
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
