use serde::{Deserialize, Serialize};

/// Benchmark manifest for freezing the issue list and evaluation criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    /// Name of the benchmark suite.
    pub name: String,
    /// List of issue identifiers to include in the corpus.
    pub issues: Vec<String>,
    /// Optional specific packages to focus verification on.
    pub verifier_packages: Vec<String>,
    /// Maximum iterations allowed per issue.
    pub max_iterations: u32,
    /// Token/cost budget cap for the entire benchmark run.
    pub budget_cap: f64,
}

/// Load the canonical beefcake-lx2o benchmark manifest.
pub fn load_beefcake_lx2o_manifest() -> BenchmarkManifest {
    BenchmarkManifest {
        name: "beefcake-lx2o".to_string(),
        issues: vec![
            // 6 Rust repair tasks
            "beefcake-j0uv".to_string(),
            "beefcake-kdiu".to_string(),
            "beefcake-onkz.1".to_string(),
            "beefcake-onkz.2".to_string(),
            "beefcake-onkz.3".to_string(),
            "beefcake-onkz.4".to_string(),
            // 6 multi-file integration/refactor tasks
            "beefcake-mmn3".to_string(),
            "beefcake-aruf".to_string(),
            "beefcake-frxr".to_string(),
            "beefcake-mtku".to_string(),
            "beefcake-p9gc".to_string(),
            "beefcake-snqx".to_string(),
            // 6 architecture/review/debug tasks
            "beefcake-vd8c".to_string(),
            "beefcake-1vll".to_string(),
            "beefcake-32xq".to_string(),
            "beefcake-5ol5".to_string(),
            "beefcake-arch-1".to_string(), // Placeholder
            "beefcake-arch-2".to_string(), // Placeholder
        ],
        verifier_packages: vec![],
        max_iterations: 6,
        budget_cap: 10.0,
    }
}
