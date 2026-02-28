//! NS-3: Deepthink Mode — JoinSet fan-out/fan-in pipeline.
//!
//! Implements a Map-Reduce multi-agent topology:
//!
//! ```text
//! Phase 1: Strategy Generation
//!   StrategyAgent → Vec<Strategy>
//!
//! Phase 2: Parallel Execution (Fan-out)
//!   JoinSet::spawn(worker_agent, strategy_i) × N concurrent tasks
//!
//! Phase 3: Judge Synthesis (Fan-in)
//!   JudgeAgent(Vec<StrategyOutcome>) → SynthesisResult
//! ```
//!
//! Hardware-aware concurrency: defaults to `LocalProviderConfig::max_parallel_workers`
//! (4 for vasp-02's HydraCoder). Semaphore-guarded to avoid overloading the
//! inference backend.
//!
//! ## Partial failure policy
//!
//! If at least one strategy succeeds, the judge runs on the successful set.
//! If all strategies fail, the pipeline returns `ModeOutcome::Failure`.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::modes::{
    errors::OrchestrationError,
    provider_config::ModeRunnerConfig,
    runner::{ModeContext, ModeRequest, ModeRunner, StepResult},
    types::{Artifact, ModeOutcome, Strategy, StrategyOutcome, StrategyResult, SynthesisResult},
};

// ── DeepthinkPhase ────────────────────────────────────────────────────────────

/// Internal phase tracker for the Deepthink pipeline.
#[derive(Debug)]
enum DeepthinkPhase {
    /// Not yet started.
    Idle,
    /// Generate N strategies from the problem statement.
    GeneratingStrategies { problem: String },
    /// Fan-out: execute strategies in parallel.
    ExecutingStrategies {
        problem: String,
        strategies: Vec<Strategy>,
    },
    /// Fan-in: judge synthesises results.
    Synthesising {
        problem: String,
        outcomes: Vec<StrategyOutcome>,
    },
    /// Terminal.
    Complete(SynthesisResult),
    /// Terminal failure.
    Failed(String),
}

// ── DeepthinkRunner ───────────────────────────────────────────────────────────

/// Implements the Deepthink mode as a `ModeRunner`.
pub struct DeepthinkRunner {
    config: ModeRunnerConfig,
    phase: DeepthinkPhase,
    /// Number of strategies to generate (defaults to `max_parallel_workers`).
    num_strategies: usize,
}

impl DeepthinkRunner {
    pub fn new(config: ModeRunnerConfig) -> Self {
        let num_strategies = config.local.max_parallel_workers;
        Self {
            config,
            phase: DeepthinkPhase::Idle,
            num_strategies,
        }
    }

    /// Override the number of parallel strategies (useful for tests).
    pub fn with_num_strategies(mut self, n: usize) -> Self {
        self.num_strategies = n.max(1);
        self
    }

    // ── Phase 1: Strategy generation ─────────────────────────────────────

    async fn generate_strategies(
        &self,
        problem: &str,
    ) -> Result<Vec<Strategy>, OrchestrationError> {
        debug!("generating {} strategies", self.num_strategies);

        let client = self
            .config
            .local_client()
            .map_err(|e| OrchestrationError::Configuration(format!("client build failed: {e}")))?;

        let agent = client
            .agent(&self.config.models.strategy)
            .preamble(
                "You are a senior software architect. Given a problem, generate N distinct, \
                mutually-exclusive implementation strategies. \
                Respond with a JSON array of strings, one per strategy. \
                Each string should be a self-contained implementation approach (1-3 sentences). \
                Output ONLY the JSON array — no commentary.",
            )
            .temperature(0.5)
            .build();

        let prompt = format!(
            "Generate exactly {n} distinct strategies to solve this problem:\n\n{problem}\n\n\
            Respond with a JSON array of exactly {n} strings.",
            n = self.num_strategies,
            problem = problem,
        );

        let raw = agent
            .prompt(&prompt)
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        parse_strategies(&raw, self.num_strategies)
    }

    // ── Phase 2: Parallel execution (fan-out) ─────────────────────────────

    async fn execute_strategies(
        &self,
        problem: &str,
        strategies: Vec<Strategy>,
    ) -> Vec<StrategyOutcome> {
        let sem = Arc::new(Semaphore::new(self.config.local.max_parallel_workers));
        let config = Arc::new(self.config.clone());
        let problem = Arc::new(problem.to_string());
        let mut join_set: JoinSet<StrategyOutcome> = JoinSet::new();

        for strategy in strategies {
            let sem = sem.clone();
            let config = config.clone();
            let problem = problem.clone();

            join_set.spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");
                let start = Instant::now();

                let result = run_strategy_worker(&config, &problem, &strategy).await;

                StrategyOutcome {
                    strategy,
                    result,
                    elapsed: start.elapsed(),
                }
            });
        }

        let mut outcomes = Vec::new();
        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(outcome) => {
                    let status = if outcome.is_success() {
                        "success"
                    } else {
                        "failure"
                    };
                    debug!(strategy = %outcome.strategy.label, status, elapsed_ms = outcome.elapsed.as_millis());
                    outcomes.push(outcome);
                }
                Err(e) => {
                    warn!(error = %e, "strategy worker panicked");
                    // Panic in a worker task — we continue with whatever succeeded.
                }
            }
        }

        outcomes
    }

    // ── Phase 3: Fan-in / Judge synthesis ─────────────────────────────────

    async fn synthesise(
        &self,
        problem: &str,
        outcomes: &[StrategyOutcome],
    ) -> Result<SynthesisResult, OrchestrationError> {
        let successful: Vec<&StrategyOutcome> =
            outcomes.iter().filter(|o| o.is_success()).collect();

        if successful.is_empty() {
            return Err(OrchestrationError::InferenceFailure(
                "all strategy workers failed — cannot synthesise".to_string(),
            ));
        }

        info!(
            total = outcomes.len(),
            successful = successful.len(),
            "synthesising from successful strategies"
        );

        let client = self
            .config
            .local_client()
            .map_err(|e| OrchestrationError::Configuration(format!("client build failed: {e}")))?;

        let agent = client
            .agent(&self.config.models.judge)
            .preamble(
                "You are a principal engineer reviewing multiple competing implementations. \
                Select the best one or synthesise a superior combined solution. \
                First identify the winning approach (or 'combined'), then output the final code. \
                Format your response as:\n\
                WINNER: <strategy label or 'combined'>\n\
                RATIONALE: <1-2 sentence explanation>\n\
                CODE:\n<final code>",
            )
            .temperature(self.config.critic_temperature)
            .build();

        let candidates: String = successful
            .iter()
            .map(|o| {
                let code = o.artifact().map(|a| a.content.as_str()).unwrap_or("");
                format!("=== {} ===\n{}", o.strategy.label, code)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Problem:\n{problem}\n\nCandidate implementations:\n\n{candidates}\n\n\
            Select the best or synthesise a superior solution."
        );

        let raw = agent
            .prompt(&prompt)
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        parse_synthesis_result(&raw, outcomes, successful.len())
    }
}

#[async_trait]
impl ModeRunner for DeepthinkRunner {
    fn name(&self) -> &'static str {
        "deepthink"
    }

    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        self.config
            .validate()
            .map_err(OrchestrationError::Configuration)?;
        self.phase = DeepthinkPhase::GeneratingStrategies {
            problem: request.task.clone(),
        };
        Ok(())
    }

    async fn step(&mut self, ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        if ctx.is_cancelled() {
            return Ok(StepResult::Failed(OrchestrationError::Cancelled(
                "cancelled".to_string(),
            )));
        }

        // Temporarily swap out the phase to take ownership.
        let current = std::mem::replace(&mut self.phase, DeepthinkPhase::Idle);

        self.phase = match current {
            DeepthinkPhase::GeneratingStrategies { problem } => {
                match self.generate_strategies(&problem).await {
                    Ok(strategies) => {
                        info!(count = strategies.len(), "strategies generated");
                        DeepthinkPhase::ExecutingStrategies {
                            problem,
                            strategies,
                        }
                    }
                    Err(e) => DeepthinkPhase::Failed(e.to_string()),
                }
            }

            DeepthinkPhase::ExecutingStrategies {
                problem,
                strategies,
            } => {
                let outcomes = self.execute_strategies(&problem, strategies).await;
                DeepthinkPhase::Synthesising { problem, outcomes }
            }

            DeepthinkPhase::Synthesising { problem, outcomes } => {
                match self.synthesise(&problem, &outcomes).await {
                    Ok(result) => DeepthinkPhase::Complete(result),
                    Err(e) => DeepthinkPhase::Failed(e.to_string()),
                }
            }

            DeepthinkPhase::Complete(ref result) => {
                let outcome = ModeOutcome::Success {
                    artifact: result.artifact.clone(),
                    iterations: 1,
                    total_tokens: None,
                };
                return Ok(StepResult::Done(outcome));
            }

            DeepthinkPhase::Failed(ref reason) => {
                return Ok(StepResult::Failed(OrchestrationError::InferenceFailure(
                    reason.clone(),
                )));
            }

            DeepthinkPhase::Idle => {
                return Ok(StepResult::Failed(OrchestrationError::Configuration(
                    "step called before prepare".to_string(),
                )));
            }
        };

        // Check if the new phase is already terminal.
        match &self.phase {
            DeepthinkPhase::Complete(_) | DeepthinkPhase::Failed(_) => Ok(StepResult::Continue(())),
            _ => Ok(StepResult::Continue(())),
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Execute a single strategy via a worker agent.
async fn run_strategy_worker(
    config: &ModeRunnerConfig,
    problem: &str,
    strategy: &Strategy,
) -> StrategyResult {
    let client = match config.local_client() {
        Ok(c) => c,
        Err(e) => return StrategyResult::Failure(e.to_string()),
    };

    let agent = client
        .agent(&config.models.worker)
        .preamble(
            "You are an expert Rust engineer implementing a specific architectural strategy. \
            Produce correct, idiomatic Rust code. \
            Respond with code only.",
        )
        .temperature(config.generator_temperature)
        .build();

    let prompt = format!(
        "Problem:\n{problem}\n\nImplementation strategy:\n{}\n\nImplement this strategy now.",
        strategy.description
    );

    match agent.prompt(&prompt).await {
        Ok(code) => StrategyResult::Success(
            Artifact::new(code)
                .with_language("rust")
                .with_iteration(strategy.index as u32),
        ),
        Err(e) => StrategyResult::Failure(e.to_string()),
    }
}

/// Parse a JSON array of strategy strings from LLM output.
fn parse_strategies(raw: &str, expected: usize) -> Result<Vec<Strategy>, OrchestrationError> {
    // Try to extract a JSON array from the response (handle markdown fences).
    let json_str = extract_json_array(raw);

    let strings: Vec<String> = serde_json::from_str(&json_str).map_err(|e| {
        OrchestrationError::ParseFailure(format!(
            "strategy list is not a JSON array of strings: {e}\nraw: {raw}"
        ))
    })?;

    if strings.is_empty() {
        return Err(OrchestrationError::ParseFailure(
            "strategy agent returned empty list".to_string(),
        ));
    }

    let strategies: Vec<Strategy> = strings
        .into_iter()
        .take(expected)
        .enumerate()
        .map(|(i, desc)| Strategy {
            label: format!("Strategy {}", (b'A' + i as u8) as char),
            description: desc,
            index: i,
        })
        .collect();

    Ok(strategies)
}

/// Parse judge output into a `SynthesisResult`.
fn parse_synthesis_result(
    raw: &str,
    all_outcomes: &[StrategyOutcome],
    successful: usize,
) -> Result<SynthesisResult, OrchestrationError> {
    // Extract WINNER / RATIONALE / CODE sections.
    let (winner_label, rationale, code) = parse_judge_sections(raw);

    let winning_strategy = all_outcomes
        .iter()
        .find(|o| {
            winner_label
                .to_ascii_lowercase()
                .contains(&o.strategy.label.to_ascii_lowercase())
        })
        .map(|o| o.strategy.clone());

    Ok(SynthesisResult {
        artifact: Artifact::new(code).with_language("rust"),
        winning_strategy,
        successful_strategies: successful,
        total_strategies: all_outcomes.len(),
        rationale,
    })
}

fn extract_json_array(raw: &str) -> String {
    // Strip markdown fences if present.
    let stripped = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find the first '[' and last ']'.
    if let (Some(start), Some(end)) = (stripped.find('['), stripped.rfind(']')) {
        stripped[start..=end].to_string()
    } else {
        stripped.to_string()
    }
}

fn parse_judge_sections(raw: &str) -> (String, String, String) {
    let mut winner = String::from("unknown");
    let mut rationale = String::new();
    let mut code = String::new();
    let mut in_code = false;

    for line in raw.lines() {
        if line.starts_with("WINNER:") {
            winner = line.trim_start_matches("WINNER:").trim().to_string();
        } else if line.starts_with("RATIONALE:") {
            rationale = line.trim_start_matches("RATIONALE:").trim().to_string();
        } else if line.starts_with("CODE:") {
            in_code = true;
        } else if in_code {
            code.push_str(line);
            code.push('\n');
        }
    }

    if code.is_empty() {
        // Fallback: treat entire response as code.
        code = raw.to_string();
    }

    (winner, rationale, code)
}

// ── Public helper ─────────────────────────────────────────────────────────────

/// Run a Deepthink task with default configuration.
pub async fn run_deepthink(task: impl Into<String>) -> ModeOutcome {
    let config = ModeRunnerConfig::default();
    let mut runner = DeepthinkRunner::new(config.clone());
    let orch = crate::modes::ModeOrchestrator::new(config);
    orch.run(&mut runner, ModeRequest::new(task)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strategies_valid_json() {
        let raw = r#"["Use a Vec", "Use a HashMap", "Use a BTreeMap"]"#;
        let strategies = parse_strategies(raw, 3).unwrap();
        assert_eq!(strategies.len(), 3);
        assert_eq!(strategies[0].label, "Strategy A");
        assert_eq!(strategies[1].index, 1);
    }

    #[test]
    fn parse_strategies_strips_markdown_fence() {
        let raw = "```json\n[\"approach A\", \"approach B\"]\n```";
        let strategies = parse_strategies(raw, 2).unwrap();
        assert_eq!(strategies.len(), 2);
    }

    #[test]
    fn parse_strategies_empty_returns_error() {
        let raw = "[]";
        assert!(parse_strategies(raw, 2).is_err());
    }

    #[test]
    fn parse_judge_sections_extracts_fields() {
        let raw = "WINNER: Strategy A\nRATIONALE: Clear and simple.\nCODE:\nfn main() {}";
        let (winner, rationale, code) = parse_judge_sections(raw);
        assert_eq!(winner, "Strategy A");
        assert_eq!(rationale, "Clear and simple.");
        assert!(code.contains("fn main()"));
    }

    #[test]
    fn extract_json_array_handles_nested() {
        let raw = "Here is the array: [\"a\", \"b\"]";
        assert_eq!(extract_json_array(raw), "[\"a\", \"b\"]");
    }
}
