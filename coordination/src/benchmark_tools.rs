//! MCP Tools for Benchmark Mode
//!
//! Provides MCP tool handlers for running rust-bench benchmarks.

use crate::benchmark::problem::ProblemSubset;
use crate::benchmark::{BenchmarkConfig, BenchmarkMetrics, BenchmarkSession, Difficulty};
use crate::feedback::{Compiler, RustcErrorParser};
use crate::router::task_classifier::ModelTier;
use crate::router::{ModelRouter, ModelSelection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Shared benchmark state accessible by MCP tools
pub struct BenchmarkState {
    /// Current session (if any)
    pub session: Option<BenchmarkSession>,
    /// Model router
    pub router: ModelRouter,
    /// Configuration
    pub config: BenchmarkConfig,
}

impl BenchmarkState {
    /// Create new state with default config
    pub fn new() -> Self {
        Self {
            session: None,
            router: ModelRouter::new(),
            config: BenchmarkConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            session: None,
            router: ModelRouter::new().with_escalation_threshold(config.escalation_threshold),
            config,
        }
    }

    /// Start a new session
    pub fn start_session(&mut self) -> &BenchmarkSession {
        let session = BenchmarkSession::new(self.config.clone());
        self.session = Some(session);
        self.session.as_ref().unwrap()
    }

    /// Get current session
    pub fn current_session(&self) -> Option<&BenchmarkSession> {
        self.session.as_ref()
    }

    /// Get current session mutably
    pub fn current_session_mut(&mut self) -> Option<&mut BenchmarkSession> {
        self.session.as_mut()
    }
}

impl Default for BenchmarkState {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared benchmark state
pub type SharedBenchmarkState = Arc<RwLock<BenchmarkState>>;

/// Create a shared benchmark state
pub fn create_shared_benchmark_state(config: BenchmarkConfig) -> SharedBenchmarkState {
    Arc::new(RwLock::new(BenchmarkState::with_config(config)))
}

// ============================================================================
// Tool Request/Response Types
// ============================================================================

/// Request to start a benchmark session
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BenchmarkStartRequest {
    /// Path to rust-bench repository
    #[schemars(description = "Path to cloned rust-bench repository")]
    pub bench_path: String,

    /// Maximum correction iterations per problem
    #[schemars(description = "Max iterations for correction loop (default: 5)")]
    pub max_iterations: Option<u32>,

    /// Only run problems of this difficulty
    #[schemars(description = "Filter by difficulty: 'easy' or 'hard' (default: all)")]
    pub difficulty_filter: Option<String>,

    /// Maximum number of problems to run
    #[schemars(description = "Limit number of problems (default: all)")]
    pub limit: Option<usize>,
}

/// Response from benchmark start
#[derive(Debug, Serialize)]
pub struct BenchmarkStartResponse {
    /// Session ID
    pub session_id: String,
    /// Number of problems loaded
    pub problem_count: usize,
    /// Problems by difficulty
    pub by_difficulty: DifficultyBreakdown,
    /// Configuration summary
    pub config_summary: String,
}

/// Breakdown of problems by difficulty
#[derive(Debug, Serialize)]
pub struct DifficultyBreakdown {
    pub easy: usize,
    pub hard: usize,
}

/// Request to solve the next problem
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BenchmarkSolveNextRequest {
    /// Optional: Override the model tier to use
    #[schemars(description = "Model tier: 'fast', 'specialized', or 'reasoning'")]
    pub model_tier: Option<String>,
}

/// Response from solving a problem
#[derive(Debug, Serialize)]
pub struct BenchmarkSolveResponse {
    /// Problem ID
    pub problem_id: String,
    /// Problem difficulty
    pub difficulty: String,
    /// Whether first attempt compiled
    pub compiled_first: bool,
    /// Whether it eventually compiled (after correction)
    pub compiled_final: bool,
    /// Iterations used
    pub iterations: u32,
    /// Tokens consumed
    pub tokens: u32,
    /// Final status
    pub status: String,
    /// Progress summary
    pub progress: String,
}

/// Request to check compilation
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CompileCheckRequest {
    /// Rust code to check
    #[schemars(description = "Rust code to compile")]
    pub code: String,

    /// Optional crate name
    #[schemars(description = "Crate name (default: temp_check)")]
    pub crate_name: Option<String>,
}

/// Response from compile check
#[derive(Debug, Serialize)]
pub struct CompileCheckResponse {
    /// Whether code compiled
    pub success: bool,
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
    /// Formatted errors for LLM
    pub errors: String,
    /// Parsed error categories
    pub categories: Vec<String>,
}

/// Request to get model selection
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SelectModelRequest {
    /// Task description
    #[schemars(description = "Description of the task")]
    pub task: Option<String>,

    /// Error message to analyze
    #[schemars(description = "Compilation error message")]
    pub error: Option<String>,
}

/// Response with model selection
#[derive(Debug, Serialize)]
pub struct SelectModelResponse {
    /// Selected model tier
    pub tier: String,
    /// Model identifier
    pub model_id: String,
    /// Recommended temperature
    pub temperature: f32,
    /// Recommended max tokens
    pub max_tokens: u32,
    /// Reason for selection
    pub reason: String,
}

/// Request to get benchmark status
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BenchmarkStatusRequest {
    /// Whether to include detailed metrics
    #[schemars(description = "Include detailed metrics (default: false)")]
    pub detailed: Option<bool>,
}

/// Response with benchmark status
#[derive(Debug, Serialize)]
pub struct BenchmarkStatusResponse {
    /// Session ID
    pub session_id: String,
    /// Session status
    pub status: String,
    /// Progress summary
    pub progress: String,
    /// Current problem ID
    pub current_problem: Option<String>,
    /// Detailed metrics (if requested)
    pub metrics: Option<BenchmarkMetrics>,
}

// ============================================================================
// Tool Implementations
// ============================================================================

/// Benchmark tool handler implementations
pub struct BenchmarkTools {
    state: SharedBenchmarkState,
}

impl BenchmarkTools {
    /// Create new tool handler with shared state
    pub fn new(state: SharedBenchmarkState) -> Self {
        Self { state }
    }

    /// Start a new benchmark session
    pub fn benchmark_start(
        &self,
        req: BenchmarkStartRequest,
    ) -> Result<BenchmarkStartResponse, String> {
        let mut state = self.state.write().map_err(|e| e.to_string())?;

        // Configure based on request
        state.config.max_correction_iterations = req.max_iterations.unwrap_or(5);

        if let Some(diff) = &req.difficulty_filter {
            let difficulty = match diff.to_lowercase().as_str() {
                "easy" => Some(Difficulty::Easy),
                "hard" => Some(Difficulty::Hard),
                _ => None,
            };
            if let Some(d) = difficulty {
                if let Some(ref mut subset) = state.config.subset {
                    subset.difficulty = Some(d);
                } else {
                    state.config.subset = Some(ProblemSubset {
                        problem_ids: None,
                        difficulty: Some(d),
                        limit: req.limit,
                    });
                }
            }
        }

        if req.limit.is_some() {
            if let Some(ref mut subset) = state.config.subset {
                subset.limit = req.limit;
            } else {
                state.config.subset = Some(ProblemSubset {
                    problem_ids: None,
                    difficulty: None,
                    limit: req.limit,
                });
            }
        }

        // Start session
        let session = state.start_session();
        let session_id = session.id.clone();

        // Load problems
        let session = state.current_session_mut().unwrap();
        let count = session
            .load_problems(&req.bench_path)
            .map_err(|e| format!("Failed to load problems: {}", e))?;

        // Count by difficulty
        let easy = session
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Easy)
            .count();
        let hard = session
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Hard)
            .count();

        Ok(BenchmarkStartResponse {
            session_id,
            problem_count: count,
            by_difficulty: DifficultyBreakdown { easy, hard },
            config_summary: format!(
                "max_iterations={}, escalation_threshold={}",
                state.config.max_correction_iterations, state.config.escalation_threshold
            ),
        })
    }

    /// Get benchmark status
    pub fn benchmark_status(
        &self,
        req: BenchmarkStatusRequest,
    ) -> Result<BenchmarkStatusResponse, String> {
        let mut state = self.state.write().map_err(|e| e.to_string())?;

        let session = state
            .current_session_mut()
            .ok_or("No active benchmark session")?;

        // Calculate metrics if requested
        if req.detailed.unwrap_or(false) {
            session.calculate_metrics();
        }

        Ok(BenchmarkStatusResponse {
            session_id: session.id.clone(),
            status: format!("{:?}", session.status),
            progress: session.progress_summary(),
            current_problem: session.current_problem().map(|p| p.id.clone()),
            metrics: if req.detailed.unwrap_or(false) {
                session.metrics.clone()
            } else {
                None
            },
        })
    }

    /// Check if code compiles
    pub fn compile_check(&self, req: CompileCheckRequest) -> Result<CompileCheckResponse, String> {
        let state = self.state.read().map_err(|e| e.to_string())?;

        // Create temporary crate
        let work_dir = state.config.work_dir.clone();
        let crate_name = req.crate_name.unwrap_or_else(|| "temp_check".to_string());
        let crate_dir = work_dir.join(&crate_name);

        // Setup crate structure
        std::fs::create_dir_all(&crate_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(crate_dir.join("src")).map_err(|e| e.to_string())?;

        // Write Cargo.toml
        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"
"#,
            crate_name
        );
        std::fs::write(crate_dir.join("Cargo.toml"), cargo_toml).map_err(|e| e.to_string())?;

        // Write source file
        std::fs::write(crate_dir.join("src").join("lib.rs"), &req.code)
            .map_err(|e| e.to_string())?;

        // Run cargo check
        let compiler = Compiler::new(&crate_dir);
        let result = compiler.check();

        // Parse errors
        let errors = RustcErrorParser::parse_cargo_messages(&result.messages);
        let categories: Vec<String> = errors.iter().map(|e| e.category.to_string()).collect();

        Ok(CompileCheckResponse {
            success: result.success,
            error_count: result.error_count(),
            warning_count: result.warnings().len(),
            errors: result.format_for_llm(),
            categories,
        })
    }

    /// Get recommended model for a task
    pub fn select_model(&self, req: SelectModelRequest) -> Result<SelectModelResponse, String> {
        let state = self.state.read().map_err(|e| e.to_string())?;

        let selection = if let Some(task) = &req.task {
            state.router.select_for_generation(task)
        } else if let Some(_error) = &req.error {
            // Would need to parse the error, using default for now
            ModelSelection::new(ModelTier::Fast, "Default selection")
        } else {
            ModelSelection::new(ModelTier::Fast, "No context provided")
        };

        Ok(SelectModelResponse {
            tier: selection.tier.to_string(),
            model_id: selection.model_id,
            temperature: selection.temperature,
            max_tokens: selection.max_tokens,
            reason: selection.reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_state_creation() {
        let state = BenchmarkState::new();
        assert!(state.session.is_none());
    }

    #[test]
    fn test_shared_state() {
        let state = create_shared_benchmark_state(BenchmarkConfig::default());
        {
            let mut locked = state.write().unwrap();
            locked.start_session();
            assert!(locked.current_session().is_some());
        }
        {
            let locked = state.read().unwrap();
            assert!(locked.current_session().is_some());
        }
    }

    #[test]
    fn test_select_model_response() {
        let state = create_shared_benchmark_state(BenchmarkConfig::default());
        let tools = BenchmarkTools::new(state);

        let req = SelectModelRequest {
            task: Some("implement async trait with lifetime bounds".to_string()),
            error: None,
        };

        let response = tools.select_model(req).unwrap();
        // Complex task should get specialized or reasoning tier
        assert!(response.tier == "specialized" || response.tier == "reasoning");
    }
}
