# Self-Improving Swarm Architecture

> **Date:** 2026-04-12
> **Status:** Design document — implementation tracked in beads

## Problem Statement

The swarm experienced three cascading failures over Apr 7-10 that required manual intervention:

1. **Poisoned feedback** — 36K false negatives from infrastructure crashes entered TZ
2. **TS lock-in** — Thompson Sampling converged on a single variant (100% omnicoder)
3. **Hung model** — The winner variant hung, causing 100% worker delegation failure

None of these were detected or corrected automatically. The swarm needs a self-improvement loop.

## Architecture: Three-Layer Defense

### Layer 1: Health-Gated Feedback

**Problem:** The swarm posted `task_resolved=false` feedback for every crash, even when the failure was infrastructure (not model quality). This poisoned TS.

**Solution:** Before posting feedback to TZ, verify that the failure was due to model behavior, not infrastructure:

```rust
// In handle_outcome(), before post_episode_feedback():
let should_post_feedback = match &failure_reason {
    // Infrastructure failures — don't blame the model
    FailureReason::TokioPanic => false,
    FailureReason::EndpointDown => false,
    FailureReason::ContextOverflow => false,
    FailureReason::EmptyResponse => false,  // model hung, not model quality
    FailureReason::Timeout => false,
    // Model quality failures — legitimate signal
    FailureReason::VerifierFailed => true,
    FailureReason::MaxIterations => true,
    FailureReason::CircuitBreaker => true,
    // Success — always post
    FailureReason::None => true,  // resolved successfully
};

if should_post_feedback {
    post_episode_feedback(...).await;
} else {
    warn!("Skipping TZ feedback: infrastructure failure, not model quality");
}
```

**Expected impact:** Prevents future data poisoning. TZ only receives signal about actual model performance.

### Layer 2: Pre-Flight Model Health Probes

**Problem:** Thompson Sampling routed 100% to omnicoder_9b, which was hung. No fallback.

**Solution:** Before each worker delegation, probe the target model's health. If it doesn't respond within 2s, mark it as unhealthy and route to the next variant.

```rust
// Before calling the worker model:
async fn probe_model_health(endpoint: &str) -> bool {
    let client = reqwest::Client::new();
    match tokio::time::timeout(
        Duration::from_secs(2),
        client.get(format!("{endpoint}/health")).send()
    ).await {
        Ok(Ok(resp)) if resp.status().is_success() => {
            // Also test with a tiny completion to verify not hung
            let test = client.post(format!("{endpoint}/v1/chat/completions"))
                .json(&json!({"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}))
                .send().await;
            test.is_ok()
        }
        _ => false,
    }
}
```

**Integration:** Add to `cluster_health.rs` and check before each `route_to_coder()` call. Unhealthy models are temporarily excluded from the routing pool.

### Layer 3: Periodic Self-Assessment & TZ Communication

**Problem:** No automated analysis of whether the swarm is improving or degrading.

**Solution:** A background task that runs every N issues (e.g., every 10 completed issues) that:

1. **Queries TZ for recent performance** — success rate by variant over last 24h
2. **Detects anomalies** — if success rate drops >20% from baseline, flag it
3. **Identifies stuck experiments** — if one variant has >90% of traffic, force exploration
4. **Reports to beads** — creates a diagnostic issue if something looks wrong
5. **Adjusts weights** — if a variant's recent success rate is 0% on >20 episodes, reduce its weight

```
Every 10 completed issues:
  → Query TZ: SELECT variant_name, COUNT(*), AVG(value::int) FROM feedback GROUP BY variant
  → If any variant has 0% rate on >20 episodes: reduce weight to 0.05
  → If overall rate drops >20% from 7-day average: create bd issue "swarm:degradation-detected"
  → If one variant has >85% of traffic: log warning, consider adding epsilon exploration
  → Post diagnostic summary to .swarm-telemetry.jsonl
```

### Layer 4: GEPA Prompt Optimization (Future)

TZ's GEPA algorithm can automatically optimize prompt templates by:
1. Sampling prompt variations
2. Running evaluations against metrics
3. Analyzing what works/fails via LLM
4. Mutating prompts based on analysis

This requires:
- Configured inference evaluations (we have `worker_behavior_quality` and `fixer_behavior_quality`)
- A dataset of successful/failed episodes
- An LLM for analysis (cloud manager)

This is the most ambitious layer and should come after Layers 1-3 are stable.

## Implementation Status

| Layer | PR | Status | Validated? |
|-------|-----|--------|-----------|
| **1: Health-gated feedback** | [#144](https://github.com/TheFermiSea/beefcake-swarm/pull/144) | Implemented | **NOT YET VALIDATED** — needs deployment and verification that infra failures skip feedback while successes still post. Edge cases to test: what about partial failures (1 iteration, then crash at 25s)? Current threshold is `iterations==0 OR wall_time<30s`. |
| **2: Model health probes** | [#145](https://github.com/TheFermiSea/beefcake-swarm/pull/145) | Implemented | **NOT YET VALIDATED** — needs deployment and testing with a deliberately hung model. Concerns: (a) 10s probe timeout adds latency to every worker delegation, (b) probe uses "probe" as model name which may behave differently across TZ routing vs direct, (c) the probe itself could get stuck if the model accepts the connection but never responds. |
| **3: Periodic self-assessment** | In progress | Implementation below | N/A |
| **4: GEPA prompt optimization** | In progress | Implementation below | N/A |

### Validation Plan for Layers 1-2

Before declaring Layers 1-2 production-ready:

1. **Layer 1 validation:**
   - Deploy to ai-proxy
   - Deliberately crash the swarm (e.g., kill TZ gateway mid-run)
   - Verify TZ receives NO new false-negative feedback during the crash period
   - Run a successful issue and verify TZ DOES receive the positive feedback
   - Check edge: kill the model mid-iteration (iteration=1, wall_time=45s) — should this post feedback?

2. **Layer 2 validation:**
   - Deploy to ai-proxy
   - Deliberately hang a model (send it an infinite-generation request)
   - Verify the deep probe detects it within 10s and marks tier as Down
   - Verify subsequent routing skips the hung tier
   - Measure probe latency overhead on healthy models (target: <100ms)
   - Test: what happens when the probe itself hangs? (reqwest timeout should handle this)

## Implementation Priority

| Layer | Effort | Impact | Dependency |
|-------|--------|--------|------------|
| **1: Health-gated feedback** | 2 hours | Prevents all future data poisoning | None |
| **2: Model health probes** | 4 hours | Prevents hung-model cascading failure | None |
| **3: Periodic self-assessment** | 1 day | Detects degradation automatically | Layer 1 |
| **4: GEPA prompt optimization** | 3 days | Autonomous prompt improvement | Layers 1-3, TZ Autopilot API key |

## Key Design Principles

1. **Fail open, not closed** — if the self-assessment system fails, the swarm continues normally
2. **Tag everything** — every feedback record includes infrastructure health state
3. **Conservative corrections** — reduce bad variant weights, don't remove them entirely
4. **Human escalation** — create beads issues for problems that need human judgment
5. **Observable** — all self-improvement actions logged to `.swarm-telemetry.jsonl`
