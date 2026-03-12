# Phase 4: Live Observability & Token Accounting

## Inspiration Context
Symphony uses `tracker/memory.ex`, `log_file.ex`, and `observability_pubsub.ex` to power a beautiful, real-time LiveView dashboard. We will replicate this in Rust using `tokio::sync::broadcast`, `tracing`, and `axum`.

## Agent Instructions

### 1. Token and Cost Tracker
Create `coordination/src/analytics/tracker.rs`.

- Rig provides usage statistics in its `CompletionResponse`.
- Create an Actor `TokenTracker` that listens to `SwarmEvent::LLMCompletion(Usage)`.
- Accumulate `prompt_tokens`, `completion_tokens`, and calculate costs based on the specific model used (e.g., Claude 3.5 Sonnet vs Qwen).
- Store this locally in `.beefcake/tracker.json`.

### 2. Setup the PubSub Event Bus
Create `coordination/src/events/pubsub.rs`.
Use `tokio::sync::broadcast::channel(1024)`.
Every time an agent thinks, uses a tool, or changes state, broadcast a `SseEvent`.

### 3. Build the Axum SSE Dashboard
Add `axum` and `tokio-stream` to `coordination/Cargo.toml`.
Create `coordination/src/dashboard/server.rs`.

- Expose a `GET /stream` endpoint that yields Server-Sent Events from the broadcast channel.
- Expose a `GET /` endpoint that serves a simple HTML/JS page.
- The Javascript should listen to the `/stream` and append to a live terminal-like UI on the webpage, updating the Token Cost, Current State, and Active Agent dynamically.

### 4. File-based Log Streaming
Replicate Symphony's `log_file.ex`. Ensure every task creates a unique markdown log file in `.beefcake/logs/task_<id>.md`.

Write tool inputs and outputs cleanly into this markdown file in real-time, creating a persistent, readable "Receipt" of everything the swarm did to accomplish the task.

## How to Use These Files

1. Create a branch in `beefcake-swarm`.
2. Feed these files to your agent (Cursor, Claude, or Devin).
3. Prompt: "We are doing a massive architectural refactor of beefcake-swarm. Read `00_MASTER_ARCHITECTURE_PLAN.md` to understand the goal. We will execute these phases one by one. Let's start strictly with Phase 1 by reading `01_STATE_AND_ORCHESTRATION.md` and implementing it."
4. The heavy reliance on Rig matches perfectly with Rig's `AgentBuilder` paradigms, and modeling after Symphony's concepts will drastically increase the safety, visibility, and reliability of your swarm.
