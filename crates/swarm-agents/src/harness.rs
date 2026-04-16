//! First-class harness policy for swarm runs.
//!
//! The swarm already has most of the building blocks described in the harness
//! engineering literature, but they were spread across prompt text, packet
//! assembly, telemetry, and validation code. This module makes those concerns
//! explicit so runtime decisions can be traced back to a concrete harness layer.

use coordination::work_packet::types::ConstraintKind;
use coordination::{Constraint, SwarmTier, WorkPacket};
use serde::{Deserialize, Serialize};

const COMPACT_PROMPT_THRESHOLD_TOKENS: usize = 8_000;
const FANOUT_TOKEN_THRESHOLD_TOKENS: usize = 4_500;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessPolicy {
    pub economics: EconomicsLayer,
    pub perception: PerceptionLayer,
    pub guardrails: GuardrailLayer,
    pub judgment: JudgmentLayer,
    pub topology: TopologyLayer,
    pub intent: IntentLayer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicsLayer {
    pub estimated_tokens: usize,
    pub prefer_compact_prompt: bool,
    pub allow_candidate_fanout: bool,
    pub heuristic_budget: usize,
    pub playbook_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionLayer {
    pub stable_prefix_required: bool,
    pub repo_map_enabled: bool,
    pub context_firewall_enabled: bool,
    pub scoped_tool_surface: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailLayer {
    pub executable_constraints: bool,
    pub lint_feedback_is_contract: bool,
    pub no_partial_implementations: bool,
    pub scoped_edits_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentLayer {
    pub local_validator_required: bool,
    pub external_review_required: bool,
    pub min_external_passes: usize,
    pub acceptance_gate_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyLayer {
    pub manager_delegates_only: bool,
    pub isolated_workers: bool,
    pub recursive_planning_allowed: bool,
    pub parallel_workers_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentLayer {
    pub proof_of_work_required: bool,
    pub session_logging_required: bool,
    pub artifact_capture_required: bool,
    pub reviewable_output_required: bool,
}

impl HarnessPolicy {
    pub fn derive(packet: &WorkPacket, tier: SwarmTier) -> Self {
        let objective = packet.objective.to_lowercase();
        let estimated_tokens = packet.estimated_tokens();
        let architecture_task = [
            "refactor",
            "architecture",
            "orchestr",
            "swarm",
            "agent",
            "runtime",
            "topology",
            "harness",
        ]
        .iter()
        .any(|kw| objective.contains(kw));
        let multi_file = packet.file_contexts.len() >= 5 || packet.files_touched.len() >= 5;
        let council_like = matches!(tier, SwarmTier::Council | SwarmTier::Human);
        let high_risk = architecture_task || multi_file || council_like;
        let prefer_compact_prompt = matches!(tier, SwarmTier::Worker)
            || estimated_tokens >= COMPACT_PROMPT_THRESHOLD_TOKENS;
        let allow_candidate_fanout =
            !high_risk && estimated_tokens <= FANOUT_TOKEN_THRESHOLD_TOKENS;

        Self {
            economics: EconomicsLayer {
                estimated_tokens,
                prefer_compact_prompt,
                allow_candidate_fanout,
                heuristic_budget: if high_risk { 4 } else { 3 },
                playbook_budget: if high_risk { 3 } else { 2 },
            },
            perception: PerceptionLayer {
                stable_prefix_required: true,
                repo_map_enabled: packet.repo_map.is_some(),
                context_firewall_enabled: true,
                scoped_tool_surface: true,
            },
            guardrails: GuardrailLayer {
                executable_constraints: true,
                lint_feedback_is_contract: true,
                no_partial_implementations: true,
                scoped_edits_only: true,
            },
            judgment: JudgmentLayer {
                local_validator_required: true,
                external_review_required: high_risk,
                min_external_passes: usize::from(high_risk),
                acceptance_gate_required: true,
            },
            topology: TopologyLayer {
                manager_delegates_only: true,
                isolated_workers: true,
                recursive_planning_allowed: true,
                parallel_workers_allowed: allow_candidate_fanout,
            },
            intent: IntentLayer {
                proof_of_work_required: true,
                session_logging_required: true,
                artifact_capture_required: true,
                reviewable_output_required: true,
            },
        }
    }

    pub fn apply_to_packet(&self, packet: &mut WorkPacket) {
        self.trim_context(packet);
        self.inject_constraints(packet);
        self.inject_prompt_section(packet);
    }

    pub fn prompt_section(&self) -> String {
        let compact = if self.economics.prefer_compact_prompt {
            "compact prompt budget active"
        } else {
            "full prompt budget active"
        };
        let external = if self.judgment.external_review_required {
            format!(
                "external review required (min {} pass{})",
                self.judgment.min_external_passes,
                if self.judgment.min_external_passes == 1 {
                    ""
                } else {
                    "es"
                }
            )
        } else {
            "external review advisory only".to_string()
        };

        format!(
            "## Harness Contract\n\
             - Economics: {compact}; candidate fan-out {fanout}; packet est. {tokens} tokens.\n\
             - Perception: preserve stable context prefix; prefer scoped files, repo map, and condensed worker summaries over raw logs.\n\
             - Guardrails: treat verifier/lint feedback as executable constraints; no TODOs, stubs, or partial implementations.\n\
             - Judgment: local validator is mandatory; {external}; acceptance gate remains authoritative.\n\
             - Topology: manager delegates, workers edit in isolation, parallel work only for truly independent slices.\n\
             - Intent: leave reviewable proof of work via changed files, validation evidence, and swarm artifacts.\n",
            compact = compact,
            fanout = if self.economics.allow_candidate_fanout {
                "enabled"
            } else {
                "disabled"
            },
            tokens = self.economics.estimated_tokens,
            external = external,
        )
    }

    fn trim_context(&self, packet: &mut WorkPacket) {
        cap_sections(
            &mut packet.relevant_heuristics,
            self.economics.heuristic_budget,
            "Additional heuristics were omitted to preserve prompt budget.",
        );
        cap_sections(
            &mut packet.relevant_playbooks,
            self.economics.playbook_budget,
            "Additional playbooks were omitted to preserve prompt budget.",
        );
    }

    fn inject_constraints(&self, packet: &mut WorkPacket) {
        push_constraint(
            &mut packet.constraints,
            ConstraintKind::Performance,
            "Prefer append-only context updates and compact prompts when prompt size grows.",
        );
        push_constraint(
            &mut packet.constraints,
            ConstraintKind::Custom,
            "Treat verifier, lint, and reviewer feedback as executable constraints, not optional advice.",
        );
        push_constraint(
            &mut packet.constraints,
            ConstraintKind::Custom,
            "Do not leave TODOs, stubs, placeholder implementations, or partial fixes in code.",
        );
        push_constraint(
            &mut packet.constraints,
            ConstraintKind::Custom,
            "Keep edits scoped to the intended files unless the harness explicitly expands scope.",
        );
        push_constraint(
            &mut packet.constraints,
            ConstraintKind::Custom,
            "Preserve manager-worker isolation: managers plan and verify, workers execute changes.",
        );
    }

    fn inject_prompt_section(&self, packet: &mut WorkPacket) {
        let section = self.prompt_section();
        if !packet
            .relevant_heuristics
            .iter()
            .any(|item| item == &section)
        {
            packet.relevant_heuristics.insert(0, section);
        }
    }
}

fn push_constraint(constraints: &mut Vec<Constraint>, kind: ConstraintKind, description: &str) {
    if constraints.iter().any(|c| c.description == description) {
        return;
    }
    constraints.push(Constraint {
        kind,
        description: description.to_string(),
    });
}

fn cap_sections(items: &mut Vec<String>, max_items: usize, truncation_note: &str) {
    if items.len() <= max_items {
        return;
    }

    items.truncate(max_items);
    items.push(format!("_{}_", truncation_note));
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn packet(objective: &str, tier: SwarmTier) -> WorkPacket {
        WorkPacket {
            bead_id: "bead-1".into(),
            branch: "swarm/bead-1".into(),
            checkpoint: "abc123".into(),
            objective: objective.into(),
            files_touched: vec![],
            key_symbols: vec![],
            file_contexts: vec![],
            verification_gates: vec![],
            failure_signals: vec![],
            constraints: vec![],
            iteration: 1,
            target_tier: tier,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            iteration_deltas: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 200,
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
            validator_feedback: vec![],
            change_contract: None,
            repo_map: None,
            failed_approach_summary: None,
            dependency_graph: None,
        }
    }

    #[test]
    fn architecture_refactors_require_external_review() {
        let packet = packet(
            "Refactor the swarm architecture and manager topology",
            SwarmTier::Council,
        );
        let policy = HarnessPolicy::derive(&packet, SwarmTier::Council);
        assert!(policy.judgment.external_review_required);
        assert_eq!(policy.judgment.min_external_passes, 1);
        assert!(!policy.economics.allow_candidate_fanout);
    }

    #[test]
    fn applying_policy_injects_contract_and_constraints() {
        let mut packet = packet("Fix a simple parser bug", SwarmTier::Worker);
        let policy = HarnessPolicy::derive(&packet, SwarmTier::Worker);
        policy.apply_to_packet(&mut packet);
        assert!(!packet.constraints.is_empty());
        assert!(packet
            .relevant_heuristics
            .first()
            .is_some_and(|item| item.contains("## Harness Contract")));
    }
}
