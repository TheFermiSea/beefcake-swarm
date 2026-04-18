# SOTA Coding-Agent Architecture Decision — 2026-04-18

**Authors:** Research synthesis from four parallel investigations (Claude-native options, framework ecosystem, coding-specific orchestrators, academic + prior-notebook).
**Status:** Decision document. Supersedes multi-tier escalation strategy.
**Related:** `agent-harness-survey.md` (2026-03-07), `sota-multi-agent-harness-2026-04.md` (2026-04-10), `self-improving-swarm-architecture.md` (2026-04-12).

## TL;DR

Replace the inner orchestration loop of beefcake-swarm (~40–50k LOC across `driver.rs`, `state_machine.rs`, `acceptance.rs`, `reformulation.rs`, `runtime_adapter.rs`, `subtask.rs`, `pipeline.rs`, `autopilot.rs`, and dormant `coordination/src/{ensemble,council,feedback,escalation,router,work_packet,debate}/`) with **mini-SWE-agent** (100 LOC Python, MIT, 74% SWE-bench Verified, used by Meta / NVIDIA / IBM / Nebius / Anyscale) as the agent-loop reference implementation.

Keep the differentiated shell: `beads_bridge`, `worktree_bridge`, `telemetry`, `cluster_health`, `tools/`, `config`, `coordination/verifier/`, `coordination/slurm/`.

Do **not** adopt a multi-agent framework (LangGraph, CrewAI, AutoGen, Mastra, Google ADK, Agent Teams). Adopt **LiteLLM** as the model-routing component, replacing `CloudFallbackMatrix` and most of `runtime_adapter.rs`.

## Why this changes

Nine independent academic results published between May 2025 and April 2026 converged on the same empirical finding: **minimal scaffolding + strong model ≥ elaborate orchestration.** The thesis is no longer a contrarian blog take; it is replicated and peer-reviewed.

| Paper | Finding |
|---|---|
| Xia et al. **Live-SWE-agent** (arXiv 2511.13646) | 77.4% on SWE-bench Verified — single self-evolving agent built on mini-SWE-agent. Beats every hand-designed multi-agent system. |
| Jiang et al. **Putting It All into Context** (arXiv 2505.08120) | Zero scaffold, zero tools: Gemini-2.5-Pro → 50.8% SWE-bench Verified. Unscaffolded model beats tuned agent scaffolds on the same model. |
| Liu **ManagerWorker** (arXiv 2603.26458) | Hierarchical delegation only helps with genuine capability asymmetry. Weak-manager + weak-worker = WORSE than a single weak agent. |
| Song **Cross-Context Verification** (arXiv 2603.21454) | Multi-tier review (Worker→Verifier→Director) produces 100% sycophantic confirmation. Exactly what beefcake's Council tier did (5× worse). |
| Alonso et al. **TDAD** (arXiv 2603.17973) | Prescriptive TDD instructions increased regressions from 6% to 9.94%. Prescribing workflow hurts. |
| Zhang et al. **Guardrails Beat Guidance** (arXiv 2604.11088) | 5,000-run ablation: random rules help as much as expert rules; **positive directives hurt**, only negative constraints help. |
| Orogat et al. **MAFBench** (arXiv 2602.03128) | Framework choice alone: −30% planning accuracy, 90% → <30% coordination, 100× latency overhead. |
| Tripathy et al. **SWEnergy** (arXiv 2512.09543) | On weak local models, framework complexity burns 9.4× more energy for near-zero resolution gain. |
| Lindenbauer et al. **The Complexity Trap** (arXiv 2508.21433) | Observation masking halves cost and matches LLM summarization on solve rate. Fancy context-management not earning its keep. |

## Beefcake-swarm's current state

- ~124k LOC total (65k `swarm-agents` + 59k `coordination`); inner-loop replaceable portion ~8k; dead subsystems ~40k.
- Dominant failure modes per `docs/DOGFOOD_DIAGNOSIS_2026-03-16.md`: "Lazy Qwen" tool-call parameter dropping, Qwen3.5-397B Q4_K_XL premature-EOS, context overflow at `--parallel 2 / 32k ctx`. These are model/ACI problems, not orchestration problems.
- April 2026 emergency patches (commits 7247e76, eb4d7d6, 19543ab, 42c23b5, f31d430) were all damage control against our own infrastructure — disabling Council after 5× regression, broadening sandbox-error detection, lowering `max_retries`.
- Separately identified: **NotebookLM auth was broken from 2026-04-11 through 2026-04-18** (79 failed KB calls across 7 days of dogfood runs; fixed by upgrading `notebooklm-mcp-cli` 0.5.5 → 0.5.26). `query_kb_with_failsafe` silently returned empty strings on every failure, so the manager proceeded without architectural context and without debugging KB enrichment for a week. Significant compounding factor in the resolve-rate decline.

## Candidate evaluation

### Rejected frameworks

| Framework | Verdict | Why |
|---|---|---|
| LangGraph | Avoid | Aimed at stateful business workflows; rewriting our Rust state machine as a DAG gains nothing |
| CrewAI | Avoid | "Agents forget state between runs and files vanish"; no native file persistence |
| AutoGen v0.4 | Avoid | In maintenance mode; migration target is Microsoft Agent Framework (Azure-heavy) |
| Mastra | Avoid (unless TS) | TypeScript stack mismatch for a Rust-first team |
| Google ADK | Avoid | Vertex gravity; SLURM topology mismatch |
| Claude Code Agent Teams | Avoid | Experimental; interactive-only; no headless/issue-queue use case; Claude-only |
| OpenAI Agents SDK | Evaluate later | Strong long-horizon harness but handoff model fights our "manager + workers" design; OpenAI-first |
| PydanticAI | Evaluate later | Closest Python match philosophically; BYO coding tools; if we ever leave Rust |

### Adopted

**mini-SWE-agent** (github.com/SWE-agent/mini-swe-agent)
- 100 LOC Python, MIT, 74% SWE-bench Verified.
- Production users: Meta, NVIDIA, IBM, Nebius, Anyscale.
- LiteLLM backend supports every endpoint we own (Anthropic via CLIAPIProxy + llama.cpp on vasp-01/02/03).
- Bus factor near zero at this size — we become the maintainers after a week.

**LiteLLM** (github.com/BerriAI/litellm) as a component
- Replaces `CloudFallbackMatrix` and cascade logic in `runtime_adapter.rs`.
- YAML config for our 4-model cloud fallback + 3 local tiers.

**OpenHands SDK** (arXiv 2511.03690) held in reserve as Phase 3 fallback
- Peer-reviewed architecture, 72.8% SWE-bench Verified, Apache licensed.
- `LocalConversation` runs in-process; Docker is optional.
- Use if mini-SWE-agent's minimal `subprocess.run` tool surface can't handle our multi-file refactors.

## Migration plan

**Phase 1 — Spike (1–2 days)** *(this document)*
- Fork mini-SWE-agent into `python/swarm_worker.py`.
- Wire LiteLLM for our 4-model cloud cascade + 3 local endpoints.
- Run 10 representative beads issues in worktrees.
- Measure resolve rate against the current swarm on the same issues.
- Ship-or-no-ship decision gate.

**Phase 2 — Hybrid deployment (1–2 weeks)**
Rust dispatcher keeps:
- `beads_bridge`, `worktree_bridge`, `telemetry`, `cluster_health`, `tools/`, `config`
- `coordination/verifier/` (runs as the quality gate after each iteration)
- `coordination/slurm/`, `shell_safety.rs`

Python worker (~300–500 new lines):
- mini-SWE-agent inner loop + verifier tool + LiteLLM config
- JSON-on-stdin / JSON-on-stdout contract with the Rust dispatcher

Delete (~40–50k LOC):
- `driver.rs`, `state_machine.rs`, `acceptance.rs`, `reformulation.rs`, `runtime_adapter.rs`, `subtask.rs`, `pipeline.rs`, `context_firewall.rs`, `map_elites.rs`, `mutation_archive.rs`
- `coordination/src/{ensemble, feedback, escalation, council, router, work_packet, debate, speculation}/`

**Phase 3 — Graduate if needed**
Swap the Python worker for OpenHands SDK `LocalConversation` if mini-SWE-agent is too minimal. Same JSON contract; Rust shell unchanged.

## What we gain

- Alignment with published empirical results (single-agent-with-ACI-refinement, not multi-agent-with-more-tiers).
- A proven inner loop used by FAANG-scale orgs.
- 70–80% reduction in maintenance burden.
- Clarity about which parts of the codebase are ours and defensible (beads + SLURM + multi-language verifier + cluster health + telemetry) vs. commodity (the inner loop).

## What we lose

- Rust type safety at the orchestration layer (remains at the shell).
- The Council / Ensemble / Strategist / MAP-Elites / mutation-archive / reformulation experiments that have never shown measurable lift and, per peer-reviewed ablations, are probably counterproductive on our model mix.

## Explicitly NOT affected

- Beads issue tracking and native messaging
- SLURM lifecycle and cluster health
- Multi-language verifier (Rust / Python / TypeScript / Go)
- TensorZero telemetry
- NotebookLM knowledge base integration (project_brain / debugging_kb loop)
- Git worktree isolation
- Cloud proxy (CLIAPIProxy)

## Key references

Papers: arXiv 2511.13646, 2505.08120, 2603.26458, 2603.21454, 2603.17973, 2604.11088, 2602.03128, 2512.09543, 2508.21433, 2506.03011, 2511.03690, 2507.23370, 2507.19942, 2405.15793.

Adopted components: github.com/SWE-agent/mini-swe-agent, github.com/BerriAI/litellm.

Held in reserve: github.com/All-Hands-AI/OpenHands.

Reference reading: simonwillison.net/2025/… (agentic loops), newsletter.pragmaticengineer.com (How Codex is built), blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop.
