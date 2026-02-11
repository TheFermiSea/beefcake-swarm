//! Integration tests for the Escalation Engine
//!
//! Tests the Escalation Engine with real VerifierReports from actual
//! broken crates, validating the full classify → decide → escalate flow.

use rust_cluster_mcp::escalation::engine::{EscalationEngine, SuggestedAction};
use rust_cluster_mcp::escalation::state::{EscalationState, SwarmTier};
use rust_cluster_mcp::feedback::error_parser::ErrorCategory;
use rust_cluster_mcp::verifier::{Verifier, VerifierConfig};

/// Create a temp crate with the given lib.rs content
fn create_temp_crate(lib_content: &str) -> tempfile::TempDir {
    let temp = tempfile::TempDir::new().expect("Failed to create temp dir");
    std::fs::write(
        temp.path().join("Cargo.toml"),
        r#"[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(temp.path().join("src/lib.rs"), lib_content).unwrap();
    temp
}

/// Run the verifier on a temp crate
async fn verify_crate(temp: &tempfile::TempDir) -> rust_cluster_mcp::verifier::VerifierReport {
    let config = VerifierConfig {
        check_fmt: false,
        check_clippy: false,
        check_compile: true,
        check_test: false,
        comprehensive: false,
        ..Default::default()
    };
    let verifier = Verifier::new(temp.path(), config);
    verifier.run_pipeline().await
}

/// Test: A clean crate resolves on first iteration
#[tokio::test]
async fn test_clean_crate_resolves_immediately() {
    let temp = create_temp_crate("pub fn add(a: i32, b: i32) -> i32 { a + b }");
    let report = verify_crate(&temp).await;

    let engine = EscalationEngine::new();
    let mut state = EscalationState::new("test-clean");

    let decision = engine.decide(&mut state, &report);

    assert!(decision.resolved, "Clean crate should resolve");
    assert!(!decision.stuck);
    assert_eq!(state.total_iterations, 1);
}

/// Test: Type mismatch error stays at Implementer on first occurrence
#[tokio::test]
async fn test_type_error_stays_at_implementer() {
    let temp = create_temp_crate(r#"pub fn broken() -> String { 42 }"#);
    let report = verify_crate(&temp).await;

    assert!(!report.all_green, "Should have errors");

    let engine = EscalationEngine::new();
    let mut state = EscalationState::new("test-type-err");

    let decision = engine.decide(&mut state, &report);

    assert!(!decision.resolved);
    assert!(!decision.escalated, "First type error should not escalate");
    assert_eq!(decision.target_tier, SwarmTier::Implementer);
    assert!(matches!(decision.action, SuggestedAction::Continue));
}

/// Test: Repeated same error class triggers escalation to Integrator
#[tokio::test]
async fn test_repeated_error_escalates_to_integrator() {
    let temp = create_temp_crate(r#"pub fn broken() -> String { 42 }"#);
    let report = verify_crate(&temp).await;

    let engine = EscalationEngine::new();
    let mut state = EscalationState::new("test-repeat-esc");

    // First iteration — no escalation
    let d1 = engine.decide(&mut state, &report);
    assert!(!d1.escalated);

    // Second iteration with same error — should escalate
    let d2 = engine.decide(&mut state, &report);
    assert!(
        d2.escalated,
        "Repeated error category should trigger escalation"
    );
    assert_eq!(d2.target_tier, SwarmTier::Integrator);
    assert!(matches!(d2.action, SuggestedAction::RepairPlan));
}

/// Test: Lifetime error is correctly classified
#[tokio::test]
async fn test_lifetime_error_classification() {
    let temp = create_temp_crate(
        r#"
pub fn broken() -> &str {
    "hello"
}
"#,
    );
    let report = verify_crate(&temp).await;

    assert!(!report.all_green);
    assert!(!report.failure_signals.is_empty());

    // Check that at least one signal has the lifetime category
    let has_lifetime = report
        .failure_signals
        .iter()
        .any(|s| s.category == ErrorCategory::Lifetime);
    assert!(
        has_lifetime,
        "Expected lifetime error classification, got: {:?}",
        report
            .failure_signals
            .iter()
            .map(|s| format!("{:?}", s.category))
            .collect::<Vec<_>>()
    );
}

/// Test: Full escalation ladder — Implementer exhaustion → Integrator → Cloud → Stuck
#[tokio::test]
async fn test_full_escalation_ladder() {
    let temp = create_temp_crate(r#"pub fn broken() -> String { 42 }"#);

    let engine = EscalationEngine::new();
    let mut state = EscalationState::new("test-full-ladder");

    // Keep feeding the same error report to exhaust budgets
    let mut iterations = 0;
    let max_safety = 20; // Prevent infinite loop

    loop {
        let report = verify_crate(&temp).await;
        let decision = engine.decide(&mut state, &report);
        iterations += 1;

        println!(
            "Iter {}: tier={}, escalated={}, stuck={}, resolved={}",
            iterations, decision.target_tier, decision.escalated, decision.stuck, decision.resolved
        );

        if decision.stuck || iterations >= max_safety {
            break;
        }
    }

    assert!(state.stuck, "Should eventually get stuck");
    assert!(!state.resolved, "Should not be resolved");
    assert!(iterations <= max_safety, "Should not hit safety limit");

    // Verify escalation history
    assert!(
        !state.escalation_history.is_empty(),
        "Should have escalation records"
    );

    // Should have passed through Integrator and Cloud
    let tiers_visited: Vec<SwarmTier> =
        state.escalation_history.iter().map(|e| e.to_tier).collect();
    assert!(
        tiers_visited.contains(&SwarmTier::Integrator),
        "Should have escalated to Integrator"
    );
    assert!(
        tiers_visited.contains(&SwarmTier::Cloud),
        "Should have escalated to Cloud"
    );

    println!("Full ladder completed in {} iterations", iterations);
    println!("Escalation history: {:?}", tiers_visited);
}

/// Test: Work packet generation from real verifier report
#[tokio::test]
async fn test_work_packet_from_real_report() {
    let temp = create_temp_crate(
        r#"
pub struct Parser {
    input: String,
}

pub fn broken() -> &str {
    "hello"
}
"#,
    );
    let report = verify_crate(&temp).await;

    let engine = EscalationEngine::new();
    let mut state = EscalationState::new("test-wp");

    let decision = engine.decide(&mut state, &report);
    assert!(!decision.resolved);

    // Generate a work packet
    let generator = rust_cluster_mcp::work_packet::WorkPacketGenerator::new(temp.path());
    let packet = generator.generate(
        "test-wp",
        "Fix the lifetime error in parser",
        decision.target_tier,
        &state,
        Some(&report),
    );

    println!("Work packet: {}", packet.summary());
    println!("Estimated tokens: {}", packet.estimated_tokens());

    // Validate packet contents
    assert_eq!(packet.bead_id, "test-wp");
    assert_eq!(packet.iteration, 2); // state has 1 iteration recorded, packet is +1
    assert!(packet.has_failures(), "Should have failure signals");

    // Should contain error history
    assert!(
        !packet.error_history.is_empty(),
        "Should have error history"
    );

    // Serializes cleanly
    let json = serde_json::to_string_pretty(&packet).expect("Packet should serialize");
    assert!(json.len() > 100, "JSON should be non-trivial");
}
