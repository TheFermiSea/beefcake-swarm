//! Golden transcript fixtures — deterministic debate transcripts for
//! pass/fail/consensus/deadlock paths to prevent behavioral regressions.
//!
//! Each fixture creates a full debate transcript and verifies the exact
//! outcome and session state.

use coordination::debate::orchestrator::{CoderOutput, ReviewerOutput};
use coordination::debate::{
    ConsensusCheck, DebateConfig, DebateOrchestrator, DebatePhase, GuardrailConfig, NextAction,
    Verdict,
};

/// A transcript step: either coder output or reviewer verdict.
enum TranscriptStep {
    Coder {
        code: &'static str,
        files: Vec<&'static str>,
        explanation: &'static str,
    },
    Reviewer {
        verdict: Verdict,
        confidence: f64,
        blocking: Vec<&'static str>,
        summary: &'static str,
    },
}

/// Run a transcript to completion and return the orchestrator.
fn run_transcript(
    config: DebateConfig,
    debate_id: &str,
    issue_id: &str,
    task: &str,
    steps: Vec<TranscriptStep>,
) -> DebateOrchestrator {
    let mut orch = DebateOrchestrator::with_config(debate_id, issue_id, task, config);
    orch.start().unwrap();

    for step in steps {
        match step {
            TranscriptStep::Coder {
                code,
                files,
                explanation,
            } => {
                let output = CoderOutput {
                    code: code.to_string(),
                    files_changed: files.into_iter().map(|s| s.to_string()).collect(),
                    explanation: explanation.to_string(),
                };
                orch.submit_code(output).unwrap();
            }
            TranscriptStep::Reviewer {
                verdict,
                confidence,
                blocking,
                summary,
            } => {
                let review = ReviewerOutput {
                    check: ConsensusCheck {
                        verdict,
                        confidence,
                        blocking_issues: blocking.into_iter().map(|s| s.to_string()).collect(),
                        suggestions: vec![],
                        approach_aligned: true,
                    },
                    summary: summary.to_string(),
                };
                orch.submit_review(review).unwrap();
            }
        }
    }

    orch
}

// ── Fixture: immediate approval ────────────────────────────────────

#[test]
fn fixture_immediate_approval() {
    let orch = run_transcript(
        DebateConfig::default(),
        "fix-001",
        "issue-fix-1",
        "Add unit test for parse_config",
        vec![
            TranscriptStep::Coder {
                code:
                    "#[test]\nfn test_parse_config() { assert!(parse_config(\"valid\").is_ok()); }",
                files: vec!["src/config_test.rs"],
                explanation: "Added unit test for parse_config happy path",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Approve,
                confidence: 0.95,
                blocking: vec![],
                summary: "Test looks correct and covers the happy path",
            },
        ],
    );

    assert!(orch.is_complete());
    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
    assert!(outcome.consensus_reached);
    assert_eq!(outcome.rounds_completed, 1);
    assert_eq!(orch.session().rounds.len(), 1);
}

// ── Fixture: two-round fix ─────────────────────────────────────────

#[test]
fn fixture_two_round_fix() {
    let config = DebateConfig {
        max_rounds: 5,
        guardrails: GuardrailConfig {
            max_rounds: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    let orch = run_transcript(
        config,
        "fix-002",
        "issue-fix-2",
        "Implement retry with backoff",
        vec![
            TranscriptStep::Coder {
                code: "fn retry() { loop { call(); } }",
                files: vec!["src/retry.rs"],
                explanation: "Basic retry loop",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.80,
                blocking: vec!["No backoff delay", "No max retry limit"],
                summary: "Missing backoff and retry limit",
            },
            TranscriptStep::Coder {
                code: "fn retry(max: u32) -> Result<(), Error> { for i in 0..max { if call().is_ok() { return Ok(()) } sleep(2u64.pow(i)); } Err(MaxRetries) }",
                files: vec!["src/retry.rs"],
                explanation: "Added exponential backoff and max retries",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Approve,
                confidence: 0.92,
                blocking: vec![],
                summary: "Retry logic now includes backoff and limit",
            },
        ],
    );

    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
    assert_eq!(outcome.rounds_completed, 2);
    assert_eq!(orch.session().rounds.len(), 2);
}

// ── Fixture: deadlock after max rounds ─────────────────────────────

#[test]
fn fixture_deadlock_persistent_disagreement() {
    let config = DebateConfig {
        max_rounds: 3,
        guardrails: GuardrailConfig {
            max_rounds: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let orch = run_transcript(
        config,
        "fix-003",
        "issue-fix-3",
        "Refactor async pipeline",
        vec![
            TranscriptStep::Coder {
                code: "async fn pipeline() { step1().await; step2().await; }",
                files: vec!["src/pipeline.rs"],
                explanation: "Sequential async pipeline",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.75,
                blocking: vec!["Steps should run concurrently"],
                summary: "Use join! for concurrent execution",
            },
            TranscriptStep::Coder {
                code: "async fn pipeline() { join!(step1(), step2()); }",
                files: vec!["src/pipeline.rs"],
                explanation: "Made steps concurrent",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.70,
                blocking: vec!["Missing error propagation"],
                summary: "Errors from join are silently dropped",
            },
            TranscriptStep::Coder {
                code: "async fn pipeline() -> Result<()> { let (a, b) = join!(step1(), step2()); a?; b?; Ok(()) }",
                files: vec!["src/pipeline.rs"],
                explanation: "Added error propagation",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.65,
                blocking: vec!["Should use try_join!"],
                summary: "Prefer try_join! over manual propagation",
            },
        ],
    );

    let outcome = orch.outcome().unwrap();
    assert!(!outcome.is_success());
    assert!(outcome.needs_escalation());
    assert_eq!(outcome.terminal_phase, DebatePhase::Deadlocked);
    assert_eq!(orch.session().rounds.len(), 3);
}

// ── Fixture: abstain verdict → escalation ──────────────────────────

#[test]
fn fixture_abstain_triggers_escalation() {
    let orch = run_transcript(
        DebateConfig::default(),
        "fix-004",
        "issue-fix-4",
        "Implement complex macro",
        vec![
            TranscriptStep::Coder {
                code: "macro_rules! gen { ($t:ty) => { impl Gen for $t {} } }",
                files: vec!["src/macros.rs"],
                explanation: "Generic generation macro",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.6,
                blocking: vec!["Incomplete impl block"],
                summary: "Missing method bodies",
            },
            TranscriptStep::Coder {
                code: "macro_rules! gen { ($t:ty) => { impl Gen for $t { fn gen() -> Self { Default::default() } } } }",
                files: vec!["src/macros.rs"],
                explanation: "Added method body",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Abstain,
                confidence: 0.30,
                blocking: vec![],
                summary: "Cannot evaluate macro safety",
            },
        ],
    );

    assert!(orch.is_complete());
    let outcome = orch.outcome().unwrap();
    assert_eq!(outcome.terminal_phase, DebatePhase::Escalated);
    assert!(outcome.needs_escalation());
}

// ── Fixture: low confidence continues ──────────────────────────────

#[test]
fn fixture_low_confidence_approval_continues() {
    let orch = run_transcript(
        DebateConfig::default(),
        "fix-005",
        "issue-fix-5",
        "Add validation",
        vec![
            TranscriptStep::Coder {
                code: "fn validate(x: &str) -> bool { !x.is_empty() }",
                files: vec!["src/validation.rs"],
                explanation: "Basic validation",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Approve,
                confidence: 0.50, // below default min_confidence of 0.7
                blocking: vec![],
                summary: "Seems ok but not confident",
            },
        ],
    );

    // Low confidence approval should not reach consensus (confidence < 0.7)
    // So it continues to next round, but since we only gave one reviewer step
    // it stays in AwaitCoder. But if the guardrails don't trigger either,
    // the orchestrator should still be incomplete.
    //
    // Actually, with low confidence the consensus check fails, and guardrails
    // check next. Since it's round 1 and max_rounds=5, it continues.
    assert!(!orch.is_complete());
    assert_eq!(orch.next_action(), NextAction::AwaitCoder);
    assert_eq!(orch.session().current_round, 2);
}

// ── Fixture: multi-file change ─────────────────────────────────────

#[test]
fn fixture_multi_file_change() {
    let orch = run_transcript(
        DebateConfig::default(),
        "fix-006",
        "issue-fix-6",
        "Add new API endpoint",
        vec![
            TranscriptStep::Coder {
                code: "pub async fn list_users() -> Json<Vec<User>> { ... }",
                files: vec![
                    "src/handlers/users.rs",
                    "src/routes.rs",
                    "src/models/user.rs",
                    "tests/api/users_test.rs",
                ],
                explanation: "Added GET /users endpoint with model, handler, route, and test",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Approve,
                confidence: 0.88,
                blocking: vec![],
                summary: "All components present and correctly wired",
            },
        ],
    );

    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
}

// ── Fixture: verify transcript determinism ─────────────────────────

#[test]
fn fixture_deterministic_replay() {
    let make_steps = || {
        vec![
            TranscriptStep::Coder {
                code: "fn add(a: i32, b: i32) -> i32 { a + b }",
                files: vec!["src/math.rs"],
                explanation: "Addition function",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::RequestChanges,
                confidence: 0.80,
                blocking: vec!["Missing overflow check"],
                summary: "No overflow protection",
            },
            TranscriptStep::Coder {
                code: "fn add(a: i32, b: i32) -> Option<i32> { a.checked_add(b) }",
                files: vec!["src/math.rs"],
                explanation: "Added overflow check",
            },
            TranscriptStep::Reviewer {
                verdict: Verdict::Approve,
                confidence: 0.93,
                blocking: vec![],
                summary: "Overflow-safe implementation",
            },
        ]
    };

    let config = DebateConfig::default();
    let orch1 = run_transcript(
        config.clone(),
        "det-1",
        "i-det-1",
        "Add safe math",
        make_steps(),
    );
    let orch2 = run_transcript(config, "det-2", "i-det-2", "Add safe math", make_steps());

    // Both should resolve at the same round
    assert_eq!(orch1.session().rounds.len(), orch2.session().rounds.len());
    assert_eq!(orch1.session().phase, orch2.session().phase);

    let o1 = orch1.outcome().unwrap();
    let o2 = orch2.outcome().unwrap();
    assert_eq!(o1.terminal_phase, o2.terminal_phase);
    assert_eq!(o1.consensus_reached, o2.consensus_reached);
    assert_eq!(o1.rounds_completed, o2.rounds_completed);
}
