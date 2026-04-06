# Harness Config

Current harness parameters and tuning rationale.

## Worker Limits

| Parameter | Value | Env Var | Rationale |
|-----------|-------|---------|-----------|
| Max turns without write | 8 | `SWARM_MAX_TURNS_WITHOUT_WRITE` | Fires PromptCancelled and escalates if worker spends too many turns reading without editing. Calibrated for Rust; raise for large Python/Go files. |
| Max worker tool calls | 15 | `SWARM_MAX_WORKER_TOOL_CALLS` | Per-invocation cap on sequential worker tool calls. Raise for complex multi-file edits. |
| Max retries | 6 | `SWARM_MAX_RETRIES` | Total iterations per issue before giving up. |
| Max no-change | 3 | `SWARM_MAX_NO_CHANGE` | Circuit breaker: abort after N consecutive no-change iterations. |
| Cloud max retries | 3 | `SWARM_CLOUD_MAX_RETRIES` | Cloud-specific retry limit. |

## Context Management

| Parameter | Value | Env Var | Rationale |
|-----------|-------|---------|-----------|
| Prune after iteration | 3 | `SWARM_PRUNE_AFTER_ITERATION` | After N iterations, prune prompt to last 2 results + verifier output to manage context window usage. |
| Subtask timeout | 3600s | `SWARM_SUBTASK_TIMEOUT_SECS` | Wall-clock deadline per subtask worker (1 hour). |
| Cloud HTTP timeout | 300s | `SWARM_CLOUD_HTTP_TIMEOUT_SECS` | Per-request HTTP timeout for cloud API calls. |
| Local HTTP timeout | 2700s | `SWARM_LOCAL_HTTP_TIMEOUT_SECS` | Per-request HTTP timeout for local LLM calls (45 min). |

## Parallelism

| Parameter | Value | Env Var | Rationale |
|-----------|-------|---------|-----------|
| Parallel issues | 3 | `SWARM_PARALLEL_ISSUES` | One per GPU node via round-robin. |
| Concurrent subtasks | true | `SWARM_CONCURRENT_SUBTASKS` | Decompose multi-file issues into parallel subtasks. |

## Tuning History

_Updated as parameters are adjusted based on dogfood results._
