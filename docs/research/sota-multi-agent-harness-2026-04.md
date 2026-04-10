# SOTA Multi-Agent Coding Architectures: Research Brief & Integration Plan

> **Date:** 2026-04-10
> **Author:** Research synthesis (Claude Opus 4.6 + Asta + NotebookLM)
> **Purpose:** Inform beefcake-swarm orchestrator redesign from "parallel solo grinding" to genuine multi-model collaboration
> **Status:** Active reference document

## Table of Contents

- [Context](#context)
- [Answers to Research Questions](#answers-to-research-questions)
  - [Q1: Heterogeneous Multi-Model Collaboration](#q1-heterogeneous-multi-model-collaboration)
  - [Q2: MASAI-Style "Fixer Without Tools"](#q2-masai-style-fixer-without-tools)
  - [Q3: Small Critic Models for Patch Selection](#q3-small-critic-models-for-patch-selection)
  - [Q4: MCTS with Heterogeneous Models](#q4-mcts-with-heterogeneous-models)
  - [Q5: Telemetry to Reward Signals](#q5-telemetry-to-reward-signals)
  - [Q6: Optimal Pipeline Depth](#q6-optimal-pipeline-depth)
- [Architecture Patterns from Literature](#architecture-patterns-from-literature)
- [New Papers Beyond Initial Brief](#new-papers-beyond-initial-brief)
- [Codebase Integration Points](#codebase-integration-points)
- [NotebookLM Cross-Reference](#notebooklm-cross-reference)
- [Proposed Architecture](#proposed-architecture)
- [Implementation Priority](#implementation-priority)
- [Paper Index](#paper-index)

---

## Context

We are redesigning the orchestrator for beefcake-swarm, an autonomous coding swarm running on 3x V100S GPUs with heterogeneous local LLMs plus a cloud manager (Claude Opus 4.6). The current architecture is "parallel solo grinding" -- each issue goes to a single worker model. We want to shift to genuine multi-model COLLABORATION where models play to their measured strengths.

### Hardware & Model Inventory

| Tier | Node | Model | Parameters | Throughput | Measured Strengths |
|------|------|-------|------------|------------|-------------------|
| Scout/Fast | vasp-03 | GLM-4.7-Flash Q4_K_M | 30B/3B MoE | ~50 tok/s | Fast tool calling (tau2 84.7), high volume |
| Coder | vasp-01 | Qwen3.5-27B Q4_K_M | 27B dense | ~27 tok/s | Highest edit rate (2x Devstral), reliable code gen |
| Reasoning | vasp-02 | Devstral-Small-2-24B Q4_K_M | 24B dense | ~30 tok/s | Good reasoning, 10% edit rate (gets stuck in exploration loops) |
| SWE Specialist | vasp-03 | SERA-14B Q4_K_M | 14B (Qwen3 backbone) | TBD | Best edit accuracy (0.803), trained on SWE trajectories |
| Cloud Manager | ai-proxy | Claude Opus 4.6 (CLIAPIProxy) | N/A | N/A | 88% verifier pass rate, best overall but expensive |

### TensorZero Telemetry Summary (12K+ records per model)

- **Qwen3.5-27B**: Best at reliably producing code (2x higher edit rate than Devstral)
- **Devstral-24B**: Good reasoning, gets stuck in exploration loops (only 10% edit rate)
- **SERA-14B**: Best edit accuracy (0.803) despite being smallest -- trained on SWE trajectories
- **GLM-4.7-Flash**: Fast tool calling (50 tok/s), high volume scout
- **Cloud (Opus 4.6)**: 88% verifier pass rate, best overall but expensive

---

## Answers to Research Questions

### Q1: Heterogeneous Multi-Model Collaboration

**Question:** Are there papers specifically about heterogeneous multi-model collaboration where different-sized models (7B-30B) play specialized roles?

**Answer:** Yes, three directly relevant papers now exist.

| Paper | Year | Key Finding | Relevance |
|-------|------|------------|-----------|
| **OrchMAS** ([ARXIV:2603.03005](https://arxiv.org/abs/2603.03005)) | 2026 | Two-tier orchestration: orchestrator model constructs domain-aware reasoning pipeline, execution model performs each step. Explicitly supports **heterogeneous LLM integration with different capacities/costs** for flexible performance-efficiency trade-offs. Orchestrator iteratively updates pipeline based on intermediate feedback. | Maps directly to our Cloud Manager (orchestrator) -> Local Workers (execution) pattern. Adds dynamic role reallocation and prompt refinement mid-pipeline. |
| **CASTER** ([ARXIV:2601.19793](https://arxiv.org/abs/2601.19793)) | 2026 | Dual-Signal Router combining semantic embeddings + structural meta-features to estimate task difficulty. Self-optimizes through Cold Start -> Iterative Evolution paradigm, learning from routing failures via on-policy negative feedback. **Reduces inference cost by 72.4%** while matching success rates. | Drop-in replacement for our `dispatch.rs:41-89` error-driven routing. Our TensorZero telemetry IS the training data for the router. |
| **Triage** ([ARXIV:2604.07494](https://arxiv.org/abs/2604.07494)) | 2026 | Uses **code health metrics** as routing signal. Three tiers: light/standard/heavy (mirrors Haiku/Sonnet/Opus). Routes tasks to cheapest model that passes the same verification gate. Analytically derived two falsifiable conditions for cost-effective routing. | Almost exactly our tier structure (GLM-4.7-Flash / Qwen3.5-27B / Devstral-24B). Code health signals available from our verifier pipeline. |

**Critical gap:** No paper studies V100S-class hardware constraints or MoE-vs-dense routing tradeoffs. We are in novel territory with our exact hardware mix.

### Q2: MASAI-Style "Fixer Without Tools"

**Question:** Has anyone studied the MASAI-style "Fixer without tools" pattern specifically for models that are good reasoners but poor agents?

**Answer:** Partially confirmed, still a research gap.

The MASAI paper ([ARXIV:2406.11638](https://arxiv.org/abs/2406.11638)) confirms the Fixer sub-agent uses CoT without environment access -- it receives code context and outputs patches through pure reasoning:

> *"Each sub-agent is equipped with a set of actions and a corresponding strategy... This strategy can include methods such as vanilla completion, Chain-of-Thought or ReAct. Note that in MASAI, agentic behavior is confined to individual stages."*

The MASAI architecture has 5 sequential sub-agents:
1. **Test Template Generator** (ReAct) -- discovers repo testing patterns
2. **Issue Reproducer** (ReAct) -- creates reproduction test
3. **Edit Localizer** (ReAct) -- finds files/functions to modify
4. **Fixer** (CoT, NO environment access) -- generates candidate patches
5. **Ranker** (CoT) -- ranks patches by running reproduction tests

**No follow-up paper specifically studies why removing tools helps poor-agent-good-reasoner models.** This is a genuine research gap.

However, the "Inside the Scaffold" taxonomy ([ARXIV:2604.03515](https://arxiv.org/abs/2604.03515)) provides indirect evidence:
> *"Scaffold-side code understanding (AST, graph DB, SBFL) outperforms LLM-as-navigator for localization"*

This means the scaffold should do the navigation, and the LLM should do the reasoning.

**Integration implication for Devstral-24B:** Don't give it tool access for patch generation. Feed it the localized code context (from GLM-4.7-Flash scout), error diagnosis, and let it output patches via CoT only. This maps directly to our existing `reformulation.rs` intent contracts + `dispatch.rs` routing.

### Q3: Small Critic Models for Patch Selection

**Question:** Are there results on critic/reward models fine-tuned on small models (14B range) for code patch selection?

**Answer:** Strong evidence that 14B models work as critics.

| Paper | Model Size | Approach | Result |
|-------|-----------|----------|--------|
| **"Smaller Models, Smarter Rewards"** ([ARXIV:2510.23083](https://arxiv.org/abs/2510.23083)) | Phi-4 family (~14B) | Value-head regression trained on APPS benchmark | >20% improvement in search capability. "Small LLMs are capable of serving as effective reward models or code evaluation critics." |
| **Vul-R2** ([ARXIV:2510.05480](https://arxiv.org/abs/2510.05480)) | Qwen2.5-14B-Instruct | Binary critic (fix-or-not) + similarity metric | Critic score combined with ground-truth similarity as reward signal for RLVR |
| **RePair** ([ARXIV:2408.11296](https://arxiv.org/abs/2408.11296)) | <20B models | Reward model with pairwise ranking on error severity | "Process-based [approach] nearly matches the performance of closed-source commercial large-scale LMs" |
| **CodeRL** ([ARXIV:2207.01780](https://arxiv.org/abs/2207.01780)) | Various | Trained critic predicts functional correctness | Best performance with M=1 (focus on best candidate), not M=2-4 |
| **AdaPatcher** | CodeLlama-7B | Two-stage: SFT for bug localization, DPO for patch refinement | 62% consistency and 67.57% accuracy on ACPR benchmark |

**Implication for SERA-14B:** Yes, SERA-14B could serve as a patch critic. Its SWE-trajectory training (0.803 edit accuracy) gives it domain-specific patch quality understanding. We'd fine-tune a value head on our TensorZero data (verifier pass/fail outcomes) to create a regression model.

**CodeRL insight:** Best candidate selection works with M=1 (focus on one best candidate for repair), not spreading across multiple candidates. This suggests the critic should confidently select ONE top candidate, not hedge.

### Q4: MCTS with Heterogeneous Models

**Question:** Has anyone applied SWE-Search/MCTS to a heterogeneous model setup where different models handle different tree-search roles?

**Answer:** One directly relevant paper, but no heterogeneous MCTS study yet.

**TSAPR** ([ARXIV:2507.01827](https://arxiv.org/abs/2507.01827), 2025): Tree Search-based APR framework integrating MCTS into patch exploration. Achieves 164/300 on SWE-Bench-Lite. Key properties:
- Evaluate-and-improve paradigm (vs trial-and-error)
- Global assessment of candidate patches
- Multi-path exploration
- **Model-agnostic** -- compatible with various base LLMs

**No paper assigns different MCTS roles to different model sizes.** But the architecture maps naturally to our hardware:

| MCTS Role | Proposed Model | Rationale |
|-----------|---------------|-----------|
| **Expansion** (generate candidate patches) | Qwen3.5-27B (coder) | Highest reliable edit rate (2x Devstral) |
| **Simulation/Rollout** (estimate patch quality) | SERA-14B (critic) | Best edit accuracy (0.803), fast on shared GPU |
| **Evaluation** (qualitative assessment) | Devstral-24B (reasoning) | Good reasoning, bad agent -- perfect for evaluation-only |
| **Selection** (UCB policy, tree management) | Deterministic code | No LLM needed -- pure UCB1/PUCT algorithm |

SWE-Search ([ARXIV:2410.20285](https://arxiv.org/abs/2410.20285)) showed 23% relative improvement through MCTS **without larger models** -- performance scales with inference-time compute (deeper search), not model size. This is exactly what we need on V100S hardware.

### Q5: Telemetry to Reward Signals

**Question:** Are there papers on converting TensorZero-style telemetry data into reward signals for training critic models or optimizing role assignment?

**Answer:** Three papers provide the blueprint.

| Paper | Mechanism | Applicability |
|-------|-----------|--------------|
| **StepORLM** ([ARXIV:2509.22558](https://arxiv.org/abs/2509.22558)) | Co-evolutionary loop: policy model + Generative Process Reward Model (GenPRM) iteratively improve. GenPRM uses Weighted DPO. Acts as universal process verifier boosting inference scaling for any LLM. | GenPRM architecture could be applied to SERA-14B. Our mutation_archive.rs outcomes serve as the dual-feedback signal. |
| **CASTER** ([ARXIV:2601.19793](https://arxiv.org/abs/2601.19793)) | Self-optimizes through Cold Start -> Iterative Evolution, learning from routing failures via on-policy negative feedback. | Directly applicable: our TensorZero records contain (task, model, outcome) triples -- exactly what CASTER's router trains on. |
| **RAGEN/StarPO** ([ARXIV:2504.20073](https://arxiv.org/abs/2504.20073)) | Framework for trajectory-level agent RL. Key finding: **"without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerges through multi-turn RL."** | Our TensorZero data has per-tool-call traces -- exactly these fine-grained signals. Warns against coarse rewards. |

**Concrete pipeline for our data:**

```
TensorZero Records (12K+ per model)
    -> Extract (issue_id, model, tool_calls, verifier_pass, iterations)
    -> Create preference pairs: (winning_trajectory, losing_trajectory) per issue
    -> Train SERA-14B value head via DPO on trajectory pairs
    -> Use as critic for best-of-N selection and routing decisions
```

**Key warning from RAGEN:** Coarse outcome-only rewards are insufficient. We need per-step signals (which tool calls led to progress vs regression). Our TensorZero data captures this granularity.

### Q6: Optimal Pipeline Depth

**Question:** What does the literature say about optimal pipeline depth? Is there evidence of diminishing returns?

**Answer:** No formal study, but strong empirical evidence converges on 3-5 stages.

| System | Stages | Performance |
|--------|--------|-------------|
| MASAI | 5 (Template -> Reproducer -> Localizer -> Fixer -> Ranker) | 28.33% SWE-bench Lite |
| TSAPR | 3 MCTS phases (Select -> Expand+Simulate -> Backprop), iterated | 164/300 SWE-bench Lite |
| AgentScope | 3 (reproduce -> fix -> test) | 63.4% SWE-bench Verified |
| IBM iSWE-Agent | 4 (Localize -> Edit -> Test -> Judge) | 31% Multi-SWE-Bench Java |
| Agentless | 3 (Localize -> Repair -> Validate) | SOTA at "remarkably low cost" |

**Inside the Scaffold** ([ARXIV:2604.03515](https://arxiv.org/abs/2604.03515)) analyzed 13 agents:
- 11/13 compose **multiple** loop primitives
- Five composable primitives: ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search
- No single architecture dominates -- agents occupy continuous spectra

**Consensus pattern:**
```
Localize -> [optional: Reproduce] -> Generate Candidates -> Rank/Select -> Verify
```

Diminishing returns appear beyond 5 stages because inter-stage context loss accumulates. MASAI's 5 stages work because each sub-agent has independent context windows.

---

## Architecture Patterns from Literature

### Core Patterns (from initial web research, confirmed by Asta)

| Pattern | Paper | Key Mechanism | SWE-bench Score |
|---------|-------|--------------|----------------|
| **Modular Sub-Agent Pipeline** | MASAI (ICLR 2025) | 5 sequential sub-agents with distinct strategies | 28.33% Lite |
| **MCTS + Multi-Agent Debate** | SWE-Search (ICLR 2025) | Three agents with Monte Carlo Tree Search | 23% relative improvement |
| **Best-of-N + Trained Critic** | OpenHands | Solver generates N candidates, TD-learning critic evaluates | 66.4% Verified (N=5) |
| **Tournament Selection** | IBM iSWE-Agent | Fine-tuned scorer (Qwen-2.5-Coder-32B), pairwise tournament | 31% Multi-SWE-Bench Java |
| **Independent Test Designer** | AgentCoder | Test agent never sees generated code | Pass@1: 71.3% -> 79.9% |
| **Four-Agent Synthesis** | MapCoder | Retrieval -> Planning -> Generation -> Debugging | 93.9% HumanEval |
| **Meta-Harness Optimization** | Meta-Harness | LLM optimizes the harness itself with 10M token traces | 76.4% TerminalBench-2 |

### Design Patterns from Surveys

From "Designing LLM-based Multi-Agent Systems for SE Tasks" ([ARXIV:2511.08475](https://arxiv.org/abs/2511.08475)):
- 16 design patterns identified; **Role-Based Cooperation** most common
- Quality attributes by priority: Functional Suitability (94.7%) > Performance Efficiency (51.1%) > Maintainability (50.0%)

From "Multi-Agent Collaboration Mechanisms" ([ARXIV:2501.06322](https://arxiv.org/abs/2501.06322)):
- Collaboration topologies: sequential pipeline, DAG, debate/voting, hierarchical delegation

### Harness Engineering Consensus (2026)

- **"Model = CPU, Context = RAM, Harness = OS, Agent = Application"** (Phil Schmid)
- **"Build to delete"** -- design modular, since new models change optimal structure
- **"Harness as dataset"** -- competitive advantage is in captured trajectories
- **"In one test, three different harnesses running the same model scored 17 issues apart on 731 problems"** (NxCode)

---

## New Papers Beyond Initial Brief

| Paper | Year | Key Contribution | arXiv |
|-------|------|-----------------|-------|
| **Satori-SWE / EvoScale** | 2025 | **32B model matches >100B** via evolutionary test-time scaling. Selection+mutation evolutionary process. Trains model to self-evolve via RL. Most applicable to our V100S constraints. | [2505.23604](https://arxiv.org/abs/2505.23604) |
| **Wisdom and Delusion of LLM Ensembles** | 2025 | **Consensus = "popularity trap."** Diversity-based strategy captures 95% of 83% theoretical upperbound. Works in 2-model ensembles. | [2510.21513](https://arxiv.org/abs/2510.21513) |
| **EET (Early Termination)** | 2026 | Experience-driven early termination. Reduces cost 19-55% with <0.2% resolution loss. Extracts structured experience from prior executions. | [2601.05777](https://arxiv.org/abs/2601.05777) |
| **LLM Critics for Execution-Free Eval** | 2025 | LLM-based critics predict build status in 84.8% of cases without execution. F1=91.6% for edit location correctness. | [2501.16655](https://arxiv.org/abs/2501.16655) |
| **SWE-GPT (Lingma)** | 2025 | Open-source 7B/72B models fine-tuned on real code submission trajectories. 7B surpasses Llama-3.1-70B. | (via Proc. ACM SE) |
| **BOAD** | 2025 | Bandit Optimization for Agent Discovery -- auto-discovers optimal hierarchical agent architectures. | (MASAI citation) |
| **OrchMAS** | 2026 | Two-tier heterogeneous orchestration with dynamic replanning. | [2603.03005](https://arxiv.org/abs/2603.03005) |
| **CASTER** | 2026 | Context-aware task routing, 72.4% cost reduction. | [2601.19793](https://arxiv.org/abs/2601.19793) |
| **Triage** | 2026 | Code health metrics -> model tier routing. | [2604.07494](https://arxiv.org/abs/2604.07494) |
| **TSAPR** | 2025 | MCTS for automated program repair. 164/300 SWE-bench Lite. | [2507.01827](https://arxiv.org/abs/2507.01827) |
| **MHGPO** | 2025 | Critic-free RL for heterogeneous multi-agent groups. | [2506.02718](https://arxiv.org/abs/2506.02718) |
| **Smaller Models, Smarter Rewards** | 2025 | 14B models as effective code critics. | [2510.23083](https://arxiv.org/abs/2510.23083) |

---

## Codebase Integration Points

Analysis of current beefcake-swarm orchestrator architecture, mapped to redesign hooks.

### Current State Machine

```
driver.rs:
  SelectingIssue -> PreparingWorktree -> Planning -> Implementing -> Verifying
                                                                        |
                                                             (auto-fix -> Validating)
                                                                        |
                                          (escalate) <- Escalating <- Merging -> Resolved
```

### Module Integration Map

| Module | File | Current Status | Integration Hook |
|--------|------|---------------|-----------------|
| **Pipeline preflight** | `pipeline.rs:1-80` | 5-stage pure function chain | Insert critic/analyzer stage before prompt assembly |
| **Worker delegation** | `agents/manager.rs:75-86` | TensorZero function call | Patch to support multi-candidate generation per subtask |
| **Model routing** | `dispatch.rs:41-89` | Error-driven weighted scoring | Replace with CASTER-style learned router |
| **Subtask fan-out** | `subtask.rs:26-31` | JoinSet + Semaphore (prepared) | Enable for multi-candidate generation |
| **MAP-Elites archive** | `map_elites.rs:158-212` | 4x4 grid (NOT integrated) | Wire into manager for diversity-based strategy seeding |
| **Mutation archive** | `mutation_archive.rs:20-75` | Append-only JSONL (passive) | Training data source for SERA-14B critic |
| **Reformulation** | `reformulation.rs:29-64` | Intent contracts (NOT wired) | Guard against goal drift in multi-candidate pipelines |
| **Verifier** | `driver.rs:1127-1260` | Deterministic gates after each iteration | Add pre-verification LLM critic screen |
| **Stack profiles** | `config.rs:12-38` | HybridBalancedV1 default | Add new profile for MASAI-style pipeline |

### Key Existing Infrastructure

- **SwarmRole enum** (`config.rs:43-53`): `RustWorker`, `GeneralWorker`, `ReasoningWorker`, `Strategist`, `LocalManagerFallback`, `Council`
- **CoderRoute enum** (`dispatch.rs:19-27`): `RustCoder`, `GeneralCoder`, `FastFixer` -- error-driven selection
- **Routing logic** (`dispatch.rs:41-89`): Weighted scoring by ErrorCategory (borrow checker -> RustCoder, imports -> GeneralCoder, simple errors on retry -> FastFixer)
- **SubtaskPlan** (`subtask.rs:26-29`): JSON struct listing target files per worker
- **Write deadline** (`subtask.rs:83-100`): `dynamic_write_deadline()` adapts to complexity + keywords + file counts
- **PivotDecision** (`mutation_archive.rs:76-91`): iteration, rationale, new_strategy, confidence (0.0-1.0)
- **Failure classification** (`reformulation.rs:66-87`): 8 categories including `DecompositionRequired`, `ImplementationThrash`

---

## NotebookLM Cross-Reference

Existing institutional knowledge from Project Brain and Debugging KB validates and constrains the proposed architecture.

### From Project Brain

1. **Confidence-Driven Escalation Mechanisms (CDEM)** -- Tasks mapped to minimum viable compute tier, escalating only when confidence drops. Aligns with Triage paper's code-health-based routing.

2. **Ralph Wiggum Pattern** -- Strictly separates non-deterministic LLM reasoning from deterministic quality gates. Failed gates route to Specialized Fix Nodes (lint fix, type fix) rather than generic retry prompts. **The SERA-14B critic must operate as an inferential sensor, not replace the deterministic verifier.**

3. **Zero Framework Cognition (ZFC)** -- Architecture pivoting toward TOML-configured roles, prompt templates, and escalation triggers. New pipeline stages must be configurable, not hardcoded.

4. **Dynamic Turn Budgets** -- Already scale per-subtask via `dynamic_write_deadline()`. New stages must respect this budget system.

5. **Meta-Harness optimization** -- An overarching agent proposes entirely new harnesses by performing counterfactual diagnosis across 10M-token execution traces.

### From Debugging KB

6. **Worker tier (Qwen3.5-27B)**: Resolves type_mismatch in ~3 iterations but hits **"No-Change Stuck Loop"** on cross-crate refactoring -- needs earlier escalation. Confirms `SWARM_MAX_NO_CHANGE=3` calibration.

7. **Council tier**: Excels at focused 1-3 file tasks (1 iteration, ~400-560s) but needs 3-6 iterations for complex unknowns. Suggests MASAI Fixer stage should constrain Devstral to single-file reasoning.

8. **Cloud validation asymmetry**: Gemini-3-pro tends to pass validations; Claude-sonnet is stricter.

9. **Known Qwen3.5-397B Q4_K_XL bug**: Premature EOS on instruction-following format; only text-continuation works. Strategist role must use text-continuation only.

### Architectural Constraints (from NotebookLM)

- **Ralph Wiggum Pattern boundary**: SERA-14B critic is an inferential pre-filter that reduces wasted verifier runs, but deterministic verifier remains the final gate. Never let the critic override the verifier.
- **ZFC requirement**: Any new pipeline stages must be TOML-configurable from day one, not hardcoded in Rust.
- **Devstral role constraint**: Deploy as planner that decomposes cross-crate work into single-file subtasks that Qwen can handle, not as a coder.

---

## Proposed Architecture

### MASAI-Inspired Heterogeneous Pipeline

```
Phase 1: LOCALIZE (GLM-4.7-Flash scout @ 50 tok/s)
  +-- ReAct loop: navigate repo, find relevant files
  +-- Output: FileContextSet + error diagnosis
  +-- Integration: pipeline.rs stage 1, replaces current context packing

Phase 2: PLAN (Devstral-24B reasoning, CoT-only, NO tools)
  +-- Receives: FileContextSet + issue description + intent contract
  +-- Pure reasoning: generates SubtaskPlan with target files per patch
  +-- Output: Ranked list of repair strategies
  +-- Integration: reformulation.rs intent contracts

Phase 3: GENERATE (Qwen3.5-27B coder, N candidates in parallel)
  +-- subtask.rs JoinSet fan-out: 2-3 candidates per strategy
  +-- Each candidate: CoT + string-replacement edits
  +-- Devstral as "Fixer without tools" for reasoning-heavy patches
  +-- Integration: existing subtask.rs + dispatch.rs routing

Phase 4: EVALUATE (SERA-14B critic + Devstral discriminator)
  +-- SERA-14B value head: score each candidate (trained on TensorZero data)
  +-- Pre-verification screen: predict build status without execution
  +-- Diversity-based selection (NOT consensus) per "Wisdom and Delusion"
  +-- Tournament: pairwise comparison of top candidates
  +-- Integration: map_elites.rs diversity archive

Phase 5: VERIFY (Deterministic quality gates, existing verifier)
  +-- Run only top 1-2 candidates through full cargo fmt/clippy/check/test
  +-- Regression detection (driver.rs:1178-1223)
  +-- EET-style early termination: skip verification if critic confidence > threshold
  +-- Integration: acceptance.rs + driver.rs:1127-1260

Phase 6: RECORD (mutation_archive.rs + TensorZero feedback)
  +-- Record outcome: (issue, model, strategy, verifier_result, iterations)
  +-- Update MAP-Elites archive with new strategy/quality pair
  +-- Feed back to CASTER router for online learning
  +-- Integration: mutation_archive.rs + map_elites.rs
```

### GPU Scheduling Strategy

```
vasp-03 (GLM-4.7-Flash + SERA-14B shared):
  +-- Phase 1: GLM-4.7-Flash scout (fast, MoE efficient)
  +-- Phase 4: SERA-14B critic (sequential after GLM, shared GPU)

vasp-01 (Qwen3.5-27B dense):
  +-- Phase 3: Primary code generation (highest edit rate)

vasp-02 (Devstral-24B dense):
  +-- Phase 2: Planning/reasoning (CoT-only, no tools)
  +-- Phase 4: Discriminator for multi-agent debate

Cloud (Opus 4.6 via CLIAPIProxy):
  +-- Orchestrator: issue selection, strategy arbitration, escalation only
```

---

## Implementation Priority

| Rank | Action | Evidence | Expected Impact | Effort |
|------|--------|----------|----------------|--------|
| **1** | Diversity-based multi-candidate generation (2-3 candidates via `subtask.rs` fan-out) | "Wisdom & Delusion": 95% of 83% theoretical ceiling with diversity selection | **+30-50% resolution rate** | Medium -- infrastructure exists |
| **2** | SERA-14B critic trained on TensorZero preference pairs | 3 papers confirm 14B critics work; >20% search improvement | **-40% wasted verifier runs** + better ranking | Medium -- needs DPO pipeline |
| **3** | Devstral as CoT-only planner/evaluator (no tools) | MASAI Fixer pattern + our 10% edit rate telemetry | **2-3x Devstral utilization** | Low -- prompt/config change |
| **4** | CASTER-style learned router replacing `dispatch.rs` | CASTER: 72.4% cost reduction; Triage: code-health routing | **-50% cloud API spend** | High -- needs router training |
| **5** | MCTS with heterogeneous model roles | SWE-Search: 23% improvement; TSAPR: 164/300 | **Ceiling raise** | High -- full new module |

### Implementation Timeline

- **Week 1-2**: SERA-14B Critic Training -- extract preference pairs from TensorZero, fine-tune value head
- **Week 3-4**: Diversity-based multi-candidate generation -- wire `subtask.rs` fan-out + `map_elites.rs` selection
- **Week 5-6**: MASAI-style pipeline stages -- separate localize/plan/generate, wire `reformulation.rs`
- **Week 7-8**: CASTER-style learned router -- replace `dispatch.rs` heuristics
- **Month 3**: MCTS integration -- full tree search with heterogeneous model roles

---

## Paper Index

### Primary References (Asta-confirmed)

| # | Title | Authors | Year | arXiv | Venue |
|---|-------|---------|------|-------|-------|
| 1 | MASAI: Modular Architecture for Software-engineering AI Agents | Arora et al. | 2024 | [2406.11638](https://arxiv.org/abs/2406.11638) | ICLR 2025 |
| 2 | SWE-Search: Enhancing Software Agents with MCTS | Antoniades et al. | 2024 | [2410.20285](https://arxiv.org/abs/2410.20285) | ICLR 2025 |
| 3 | Inside the Scaffold: Source-Code Taxonomy of Coding Agent Architectures | Rombaut | 2026 | [2604.03515](https://arxiv.org/abs/2604.03515) | -- |
| 4 | Wisdom and Delusion of LLM Ensembles for Code Generation and Repair | Vallecillos Ruiz et al. | 2025 | [2510.21513](https://arxiv.org/abs/2510.21513) | -- |
| 5 | Triage: Routing SE Tasks to Cost-Effective LLM Tiers | Madeyski | 2026 | [2604.07494](https://arxiv.org/abs/2604.07494) | -- |
| 6 | CASTER: Context-Aware Strategy for Task Efficient Routing | Liu et al. | 2026 | [2601.19793](https://arxiv.org/abs/2601.19793) | -- |
| 7 | Satori-SWE: Evolutionary Test-Time Scaling | Zeng et al. | 2025 | [2505.23604](https://arxiv.org/abs/2505.23604) | -- |
| 8 | Smaller Models, Smarter Rewards | Groeneveld et al. | 2025 | [2510.23083](https://arxiv.org/abs/2510.23083) | -- |
| 9 | TSAPR: Tree Search Framework for Automated Program Repair | Hu et al. | 2025 | [2507.01827](https://arxiv.org/abs/2507.01827) | -- |
| 10 | EET: Experience-Driven Early Termination | Guo et al. | 2026 | [2601.05777](https://arxiv.org/abs/2601.05777) | -- |
| 11 | OrchMAS: Orchestrated Reasoning with Heterogeneous Expert Agents | Feng et al. | 2026 | [2603.03005](https://arxiv.org/abs/2603.03005) | -- |
| 12 | MHGPO: Heterogeneous Group-Based RL for LLM-based MAS | Chen et al. | 2025 | [2506.02718](https://arxiv.org/abs/2506.02718) | -- |
| 13 | LLM Critics for Execution-Free Evaluation of Code Changes | Yadavally et al. | 2025 | [2501.16655](https://arxiv.org/abs/2501.16655) | -- |
| 14 | RePair: Automated Program Repair with Process-based Feedback | Zhao et al. | 2024 | [2408.11296](https://arxiv.org/abs/2408.11296) | -- |
| 15 | Vul-R2: A Reasoning LLM for Automated Vulnerability Repair | Wen et al. | 2025 | [2510.05480](https://arxiv.org/abs/2510.05480) | -- |
| 16 | SWE-GPT: Process-Centric LM for Automated Software Improvement | Ma et al. | 2025 | -- | Proc. ACM SE |
| 17 | StepORLM: Self-Evolving Framework with Generative Process Supervision | Zhou et al. | 2025 | [2509.22558](https://arxiv.org/abs/2509.22558) | -- |
| 18 | RAGEN: Understanding Self-Evolution via Multi-Turn RL | Wang et al. | 2025 | [2504.20073](https://arxiv.org/abs/2504.20073) | -- |
| 19 | CodeRL: Mastering Code Generation through RL | Le et al. | 2022 | [2207.01780](https://arxiv.org/abs/2207.01780) | NeurIPS 2022 |
| 20 | AgentCoder: Multi-Agent Code Generation | -- | 2023 | [2312.13010](https://arxiv.org/abs/2312.13010) | -- |
| 21 | MapCoder: Four-Agent Synthesis Cycle | -- | 2024 | [2405.11403](https://arxiv.org/abs/2405.11403) | -- |

### Survey References

| Title | arXiv |
|-------|-------|
| Designing LLM-based Multi-Agent Systems for SE Tasks | [2511.08475](https://arxiv.org/abs/2511.08475) |
| Multi-Agent Collaboration Mechanisms | [2501.06322](https://arxiv.org/abs/2501.06322) |
| LLM-Based Multi-Agent Systems for SE: Literature Review | [2404.04834](https://arxiv.org/abs/2404.04834) |

### Practitioner References

| Title | Source |
|-------|--------|
| Agent Harness Engineering (Phil Schmid) | philschmid.de/agent-harness-2026 |
| Harness Engineering Complete Guide (NxCode) | nxcode.io |
| Agents are the New Microservices | dasroot.net |
| Meta-Harness (Yoonho Lee) | yoonholee.com/meta-harness |
