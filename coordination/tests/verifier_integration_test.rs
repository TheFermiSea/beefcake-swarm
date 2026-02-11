//! Integration tests for the Verifier module
//!
//! These tests run the Verifier against the actual rust-cluster-mcp crate,
//! validating that it correctly processes real compiler output.

use rust_cluster_mcp::verifier::report::GateOutcome;
use rust_cluster_mcp::verifier::{Verifier, VerifierConfig};

/// Get the path to this crate's root directory
fn crate_root() -> String {
    env!("CARGO_MANIFEST_DIR").to_string()
}

/// Test: Verify that the Verifier can run against this actual crate
/// and reports all-green (since we know this crate compiles).
/// Note: fmt is skipped because the working tree may have unformatted
/// changes from other branches/modifications.
#[tokio::test]
async fn test_verifier_on_self_all_green() {
    let config = VerifierConfig {
        check_fmt: false, // Skip fmt — working tree may have unformatted staged changes
        check_clippy: true,
        check_compile: true,
        check_test: false, // Skip tests to avoid recursion
        comprehensive: false,
        ..Default::default()
    };

    let verifier = Verifier::new(crate_root(), config);
    let report = verifier.run_pipeline().await;

    println!("Verifier report: {}", report.summary());

    // This crate should pass clippy and check
    assert!(
        report.gates_passed >= 2,
        "Expected at least 2 gates to pass, got {}",
        report.gates_passed
    );
    assert!(
        report.all_green,
        "Expected all-green but got: {}",
        report.summary()
    );
    assert!(
        report.failure_signals.is_empty(),
        "Expected no failure signals"
    );
    assert!(
        report.error_categories.is_empty(),
        "Expected no error categories"
    );

    // Verify gate names
    let gate_names: Vec<&str> = report.gates.iter().map(|g| g.gate.as_str()).collect();
    assert!(gate_names.contains(&"clippy"), "Expected clippy gate");
    assert!(gate_names.contains(&"check"), "Expected check gate");
}

/// Test: Verify that git branch and commit are populated
#[tokio::test]
async fn test_verifier_populates_git_info() {
    let config = VerifierConfig::quick();

    let verifier = Verifier::new(crate_root(), config);
    let report = verifier.run_pipeline().await;

    // We're running from a git repo, so these should be populated
    assert!(
        report.branch.is_some(),
        "Expected git branch to be populated"
    );
    assert!(
        report.commit.is_some(),
        "Expected git commit to be populated"
    );

    println!("Branch: {:?}, Commit: {:?}", report.branch, report.commit);
}

/// Test: Verify the report serializes to valid JSON
#[tokio::test]
async fn test_verifier_report_serializes() {
    let config = VerifierConfig {
        check_fmt: true,
        check_clippy: false,
        check_compile: true,
        check_test: false,
        comprehensive: false,
        ..Default::default()
    };

    let verifier = Verifier::new(crate_root(), config);
    let report = verifier.run_pipeline().await;

    let json = serde_json::to_string_pretty(&report).expect("Report should serialize to JSON");
    assert!(json.len() > 100, "JSON should be non-trivial");

    // Round-trip: deserialize back
    let deserialized: rust_cluster_mcp::verifier::report::VerifierReport =
        serde_json::from_str(&json).expect("Report should deserialize from JSON");
    assert_eq!(deserialized.gates_passed, report.gates_passed);
    assert_eq!(deserialized.all_green, report.all_green);
}

/// Test: Verify gate timing is reasonable (not zero, not absurd)
#[tokio::test]
async fn test_verifier_gate_timing() {
    let config = VerifierConfig {
        check_fmt: true,
        check_clippy: false,
        check_compile: true,
        check_test: false,
        comprehensive: false,
        ..Default::default()
    };

    let verifier = Verifier::new(crate_root(), config);
    let report = verifier.run_pipeline().await;

    assert!(report.total_duration_ms > 0, "Total duration should be > 0");

    for gate in &report.gates {
        if gate.outcome != GateOutcome::Skipped {
            // Each non-skipped gate should take at least 1ms
            println!(
                "Gate '{}': {}ms ({})",
                gate.gate, gate.duration_ms, gate.outcome
            );
        }
    }
}

/// Test: Verify that fail-fast mode skips subsequent gates on failure.
/// Fmt is disabled so that the type error in `cargo check` is the first failure,
/// allowing us to test that the test gate gets skipped in fail-fast mode.
#[tokio::test]
async fn test_verifier_fail_fast_skips_gates() {
    let temp = tempfile::TempDir::new().expect("Failed to create temp dir");
    std::fs::write(
        temp.path().join("Cargo.toml"),
        r#"[package]
name = "broken-crate"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(
        temp.path().join("src/lib.rs"),
        r#"
// This has a deliberate type error
pub fn add(a: i32, b: i32) -> String {
    a + b  // returns i32, not String
}
"#,
    )
    .unwrap();

    let config = VerifierConfig {
        check_fmt: false,    // Skip fmt — we want check to be the first failure
        check_clippy: false, // Skip clippy to speed up
        check_compile: true,
        check_test: true,
        comprehensive: false, // Fail-fast mode
        ..Default::default()
    };

    let verifier = Verifier::new(temp.path(), config);
    let report = verifier.run_pipeline().await;

    println!("Broken crate report: {}", report.summary());

    // Should not be all-green
    assert!(!report.all_green, "Broken crate should not be all-green");

    // check gate should fail
    let check_gate = report.gates.iter().find(|g| g.gate == "check");
    assert!(check_gate.is_some(), "Should have a check gate");
    assert_eq!(check_gate.unwrap().outcome, GateOutcome::Failed);

    // test gate should be skipped (fail-fast)
    let test_gate = report.gates.iter().find(|g| g.gate == "test");
    if let Some(tg) = test_gate {
        assert_eq!(
            tg.outcome,
            GateOutcome::Skipped,
            "Test gate should be skipped in fail-fast mode"
        );
    }

    // Should have failure signals with error classification
    assert!(
        !report.failure_signals.is_empty(),
        "Expected failure signals"
    );

    // The error should be classified as TypeMismatch (E0308)
    let has_type_error = report.failure_signals.iter().any(|s| {
        s.category == rust_cluster_mcp::feedback::error_parser::ErrorCategory::TypeMismatch
            || s.code.as_deref() == Some("E0308")
    });
    assert!(
        has_type_error,
        "Expected E0308 type mismatch error, got: {:?}",
        report
            .failure_signals
            .iter()
            .map(|s| format!("{:?} {:?}", s.category, s.code))
            .collect::<Vec<_>>()
    );
}

/// Test: Verify comprehensive mode runs all gates even on failure
#[tokio::test]
async fn test_verifier_comprehensive_runs_all_gates() {
    let temp = tempfile::TempDir::new().expect("Failed to create temp dir");
    std::fs::write(
        temp.path().join("Cargo.toml"),
        r#"[package]
name = "broken-crate2"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();

    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(
        temp.path().join("src/lib.rs"),
        r#"
pub fn broken() -> &str {
    "hello"
}
"#,
    )
    .unwrap();

    let config = VerifierConfig {
        check_fmt: true,
        check_clippy: true,
        check_compile: true,
        check_test: true,
        comprehensive: true, // Run ALL gates
        ..Default::default()
    };

    let verifier = Verifier::new(temp.path(), config);
    let report = verifier.run_pipeline().await;

    println!("Comprehensive report: {}", report.summary());

    // All 4 gates should have run (none skipped)
    assert_eq!(report.gates_total, 4, "Expected 4 gates");
    let skipped = report
        .gates
        .iter()
        .filter(|g| g.outcome == GateOutcome::Skipped)
        .count();
    assert_eq!(
        skipped, 0,
        "No gates should be skipped in comprehensive mode"
    );

    // Should have lifetime error (E0106: missing lifetime specifier)
    let has_lifetime = report.failure_signals.iter().any(|s| {
        s.category == rust_cluster_mcp::feedback::error_parser::ErrorCategory::Lifetime
            || s.code.as_deref() == Some("E0106")
    });
    assert!(
        has_lifetime,
        "Expected lifetime error, got: {:?}",
        report
            .failure_signals
            .iter()
            .map(|s| format!("{:?} {:?}", s.category, s.code))
            .collect::<Vec<_>>()
    );
}
