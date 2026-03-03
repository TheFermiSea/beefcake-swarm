# Swarm Execution Quality Incident Report

## Summary

The swarm is currently **available but not converging** on SE-0 tasks.  
Primary symptom: agents repeatedly return long analysis responses without applying file edits, producing:

- `No file changes after agent response — manager may not have called workers`
- repeated iteration loops
- unresolved issues despite successful infrastructure startup

This is now the dominant blocker; infrastructure/auth issues are mostly remediated.

---

## Scope and Impact

- Affected workstream: `beefcake-hx0.1.*` (SE-0 alignment tasks)
- Observed behavior: SLURM jobs complete runtime cycles but do not resolve issues
- Operational impact:
  - high token/compute burn with low code-change yield
  - delayed progression to SE-1..SE-7
  - repeated relaunches required to attempt convergence

---

## Key Symptoms (Observed in Logs)

1. Worker iterations produce prose-only responses with no filesystem diff.
2. Iterations repeat across both `general_coder` and `rust_coder`.
3. Verifier often runs on unchanged trees and cannot advance acceptance.
4. Runs remain active for long durations with no actionable patch output.

Example pattern from active run (`job 1758`):

- worker starts
- routes to `general_coder` or `rust_coder`
- response length is large (thousands of chars)
- no git diff produced
- loop continues to next iteration

---

## Root Cause Analysis

### 1) Proxy/model lane mismatch (now fixed)

Previously, `claude-sonnet-4-6` was exposed as `owned_by=antigravity`, causing:

- `auth_unavailable`
- quota/reset errors on requested model lane

This created instability and masked deeper execution issues.

### 2) Worker non-edit behavior (still open)

Even with healthy model access, workers frequently:

- reason about intended edits
- describe plans
- return without calling edit/write tools

This indicates an execution-quality gap in tool-use follow-through, not just infra failures.

### 3) Iteration policy previously tolerated no-change loops

Earlier behavior allowed repeated no-change retries without strong corrective forcing, resulting in wasted iterations.

### 4) Task/objective mismatch for planning-style SE-0 work

Some SE-0 tasks are artifact-oriented (RFC/matrix docs), but verifier gating and prompts originally pushed code-centric behavior, increasing drift and non-action responses.

### 5) Environmental churn amplified instability

Temporary `vasp-02` maintenance/outage (MPS setup) intermittently degraded local worker availability.

---

## Remediations Applied

## A) Proxy and model exposure fixes

- Upgraded `cli-proxy-api` on `ai-proxy`:
  - from `6.7.37` to `6.8.23`
- After upgrade, `claude-sonnet-4-6` is now exposed as `owned_by=anthropic`
- Verified `chat/completions` success on `claude-sonnet-4-6`

## B) Launcher hardening

`scripts/run-swarm-sandbox.slurm` and `scripts/run-swarm.sh` now include:

- model preflight checks
- auth/quota fallback logic
- Anthropic ownership enforcement
- cloud model fallback chain
- worker endpoint failover during `vasp-02` disruption

## C) Orchestrator mitigations

In `crates/swarm-agents/src/orchestrator.rs`:

- Added no-change recovery behavior:
  - council no-change -> force worker tier
  - worker no-change -> force alternate worker route next iteration
- Added strict-edit mode prompt injection after no-change events
- Added initial tier override support via env
- Added verifier gate toggles for docs/planning runs (`SWARM_SKIP_TESTS`, `SWARM_SKIP_CLIPPY`)
- Added context budget controls to reduce oversized prompt payloads

---

## Current State (as of this report)

- Infrastructure: healthy
- Proxy auth lane: healthy on Anthropic Sonnet 4.6
- Active blocker: **worker execution quality**
  - still seeing repeated no-diff cycles even after route alternation and strict-edit prompts

---

## Why This Is Serious

This failure mode is expensive and silent:

- jobs appear “running/active” in SLURM
- agent responses look substantive
- but no code/docs changes land
- issue closure velocity collapses

Without stronger safeguards, this can consume significant cluster and cloud budget without progress.

---

## Recommended Next Actions

## Immediate (high priority)

1. Add a hard `no-change` circuit-breaker:
   - stop after N no-diff iterations
   - auto-mark run as blocked with explicit reason
2. Add deterministic artifact fallback for doc-oriented issues:
   - if objective demands a specific doc path and no edits after N iterations, generate minimally valid scaffold and hand back to reviewer cycle
3. Enforce objective payload presence:
   - reject launches that omit explicit objective for planning/documentation tasks

## Near-term

4. Add telemetry counters:
   - `no_change_rate`, `tool_call_rate`, `diff_lines_per_iteration`
5. Add model capability routing:
   - route tool-heavy tasks to models with demonstrated tool-call compliance
6. Add acceptance profiles:
   - code profile vs docs profile (different verifier gate sets and success checks)

## Structural

7. Add integration test that fails if worker returns text-only responses for X iterations.
8. Add policy-level guard in manager prompt:
   - “analysis-only replies are invalid; must either edit files or emit explicit blocked code”

---

## Operational Notes

- This incident is no longer primarily an auth/proxy outage.
- The quality bottleneck is execution behavior inside worker loops.
- Further progress depends on enforcing edit-producing behavior or failing fast with deterministic fallback.
