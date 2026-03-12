# Beefcake Swarm Optimization: Master Architecture Plan

## The Goal
Transform `beefcake-swarm` into an enterprise-grade autonomous coding harness inspired by OpenAI's `Symphony` repo. While Symphony is written in Elixir, we will leverage Rust's strengths (type safety, zero-cost abstractions, fearless concurrency) and the `Rig` AI framework to achieve superior performance.

## Core Architectural Pillars to Implement

1. **The Actor-Model Orchestrator (Symphony `orchestrator.ex` -> Rust Actor)**
   - Move away from procedural loops to an asynchronous, message-driven Actor model for the Orchestrator.
   - Agents (Coder, Reviewer, Specialist) should be spawned as independent Tokio tasks communicating via `tokio::sync::mpsc` channels, emitting `SwarmEvent` messages to a central event bus.

2. **Ephemeral Workspaces (Symphony `workspace.ex` -> Rust `WorkspaceManager`)**
   - Agents must not mutate the root project directory directly until consensus is reached.
   - Implement a `WorkspaceManager` that spawns lightweight git worktrees or isolated Docker containers for every sub-task.

3. **Declarative Skills/Tools (Symphony `.codex/skills` -> Rust `SkillRegistry`)**
   - Decouple hardcoded tools from the Rust binary.
   - Implement a dynamically loaded Skill system where a "Skill" (e.g., `git-commit`, `linear-update`, `ast-grep`) is defined by a markdown file (for the prompt) and a schema, injected into the `Rig` toolset at runtime based on the agent's current state.

4. **Live Observability & Token Accounting (Symphony `status_dashboard.ex` & `tracker.ex`)**
   - Build an embedded Axum server that streams Server-Sent Events (SSE) detailing swarm state, current agent thoughts, tool execution streams, and cumulative token costs.

## Execution Order
The coding agent MUST execute the optimization in the following order:

1. Phase 1: Implement the State Machine & Actor-based Orchestrator (`01_STATE_AND_ORCHESTRATION.md`)
2. Phase 2: Implement Ephemeral Workspaces (`02_WORKSPACE_ISOLATION.md`)
3. Phase 3: Implement Declarative Skills (`03_DYNAMIC_SKILLS.md`)
4. Phase 4: Implement the Observability Dashboard & Tracker (`04_OBSERVABILITY_AND_TRACKING.md`)
