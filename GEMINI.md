# GEMINI.md - Beefcake Swarm Project Context

## Project Overview
**Beefcake Swarm** is an autonomous coding orchestration system designed to run on a high-performance computing (HPC) cluster. It leverages local LLMs served via SLURM, coordinated by a deterministic Rust-based logic layer, and escalated to cloud-based models when necessary.

The project follows a "2-agent loop" architecture (Implementer and Validator) with rigorous quality gates (compilation, linting, testing) ensuring that only high-quality, verified code is merged.

### Core Technologies
- **Rust (Workspace):** Primary language for coordination and agent orchestration.
- **MCP (Model Context Protocol):** The `coordination` crate runs as an MCP server.
- **SLURM:** Manages the lifecycle of local LLM inference jobs on GPU nodes.
- **Beads (bd):** A graph-based issue tracking system for persistent task memory.
- **Git Worktrees:** Per-task branch isolation via `worktree_bridge.rs` (raw `git worktree` commands, not Gastown).
- **Rig:** Rust library for building LLM-powered applications, used for agent orchestration.
- **RocksDB:** Persistent state storage for model ensembles and sessions.

---

## Workspace Structure
- **`coordination/`**: The deterministic logic layer (~21k LOC).
    - **`verifier/`**: Automated quality gate pipeline (fmt, clippy, check, test).
    - **`escalation/`**: Tier routing logic based on failure patterns.
    - **`ensemble/`**: Multi-model sequential execution and voting/arbitration.
    - **`council/`**: Cloud AI escalation adapter (Gemini, Claude, GPT).
    - **`slurm/`**: SLURM inference job management and health discovery.
- **`crates/swarm-agents/`**: Rig-based orchestrator for the implementer/validator loop.
- **`flywheel/`**: TypeScript/Node.js sub-project for prompt mining and task decomposition.
- **`inference/`**: SLURM job scripts, build/validate scripts for local models.
- **`infrastructure/`**: Monitoring tools (GPU dashboard, HPC watchdog).

---

## Building and Running
### Core Commands
```bash
# Build the entire workspace
cargo build --workspace

# Run tests
cargo test -p coordination
cargo test -p swarm-agents

# Run the MCP server (stdio transport)
cargo run -p coordination

# Run orchestrator (requires inference endpoints to be active)
cargo run -p swarm-agents

# Linting and Formatting
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

---

## Development Conventions
### Issue Tracking (Beads)
This project uses **bd** (beads) for issues. 
- `bd ready`: Find available work.
- `bd show <id>`: View issue details.
- `bd update <id> --status in_progress`: Claim a task.
- `bd close <id>`: Complete a task.

### Session Completion Workflow
**MANDATORY:** You must complete these steps before ending a session:
1. File issues for remaining work in `bd`.
2. Run quality gates (fmt, clippy, test).
3. `git pull --rebase`
4. `bd sync`
5. `git push`
6. Verify `git status` shows "up to date with origin".

### Escalation Ladder
1. **Implementer (14B):** 6 iterations max.
2. **Integrator (72B):** 2 consultations + repair plan if 14B repeats errors 2x or fails compile >3x.
3. **Cloud Council (Gemini/Claude/GPT):** For architecture decisions or if Integrator is stuck.
4. **Human Intervention:** Final fallback.

---

## Inference Endpoints
Computational tasks MUST go through SLURM. Do not run workloads directly on compute nodes.

| Tier | Endpoint | Model | Role |
|------|----------|-------|------|
| **Fast (14B)** | `http://vasp-02:8080` | `strand-rust-coder-14b` | Mechanic: Fast Rust-specific fixes |
| **Coder (80B)** | `http://vasp-02:8080` | `Qwen3-Coder-Next` | Implementer: Multi-file changes |
| **Reasoning (72B)** | `http://vasp-01:8081` | `or1-behemoth` | Architect: Complex reasoning |

---

## Cluster Access
- **slurm-ctl (10.0.0.5):** Controller, NFS server.
- **vasp-01 (10.0.0.20):** GPU node (72B head).
- **vasp-02 (10.0.0.21):** GPU node (14B fast).
- **vasp-03 (10.0.0.22):** GPU node (72B RPC worker).
- **ai-proxy (100.105.113.58):** External gateway.

---

## Known Issues & Notes
- **Claude Code as Root:** The Claude Code CLI (`claude -p`) enforces security restrictions and will refuse to run with the `--dangerously-skip-permissions` flag when executed as the `root` user or with `sudo`. To run automated tasks, use a non-root user (e.g., `squires` created on `ai-proxy`).
- **Test Imports:** Some integration tests in `coordination/tests/` have unresolved crate imports (referencing `rust_cluster_mcp` instead of `coordination`).
- **Dead Code:** `crates/swarm-agents/` contains structs/methods for Phase 2 that are not yet wired up.
- **NFS Layout:** Shared binaries, scripts, and endpoints are located in `/cluster/shared/`.

---


# Building with Rig

Rig is a Rust library for building LLM-powered applications with a provider-agnostic API.
All patterns use the builder pattern and async/await via tokio.

## Quick Start

```rust
use rig::completion::Prompt;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();

    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{}", response);
    Ok(())
}
```

## Core Patterns

### 1. Simple Agent
```rust
let agent = client.agent(openai::GPT_4O)
    .preamble("System prompt")
    .temperature(0.7)
    .max_tokens(2000)
    .build();

let response = agent.prompt("Your question").await?;
```

### 2. Agent with Tools
Define a tool by implementing the `Tool` trait, then attach it:
```rust
let agent = client.agent(openai::GPT_4O)
    .preamble("You can use tools.")
    .tool(MyTool)
    .build();
```
See `references/tools.md` for the full `Tool` trait signature.

### 3. RAG (Retrieval-Augmented Generation)
```rust
let embedding_model = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
let index = vector_store.index(embedding_model);

let agent = client.agent(openai::GPT_4O)
    .preamble("Answer using the provided context.")
    .dynamic_context(5, index)  // top-5 similar docs per query
    .build();
```
See `references/rag.md` for vector store setup and the `Embed` derive macro.

### 4. Streaming
```rust
use futures::StreamExt;
use rig::streaming::StreamedAssistantContent;
use rig::agent::prompt_request::streaming::MultiTurnStreamItem;

let mut stream = agent.stream_prompt("Tell me a story").await?;

while let Some(chunk) = stream.next().await {
    match chunk? {
        MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::Text(text)
        ) => print!("{}", text.text),
        MultiTurnStreamItem::FinalResponse(resp) => {
            println!("\n{}", resp.response());
        }
        _ => {}
    }
}
```

### 5. Structured Extraction
```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, JsonSchema)]
struct Person {
    pub name: Option<String>,
    pub age: Option<u8>,
}

let extractor = client.extractor::<Person>(openai::GPT_4O).build();
let person = extractor.extract("John is 30 years old.").await?;
```

### 6. Chat with History
```rust
use rig::completion::Chat;

let history = vec![
    Message::from("Hi, I'm Alice."),
    // ...previous messages
];
let response = agent.chat("What's my name?", history).await?;
```

## Agent Builder Methods

| Method | Description |
|--------|-------------|
| `.preamble(str)` | Set system prompt |
| `.context(str)` | Add static context document |
| `.dynamic_context(n, index)` | Add RAG with top-n retrieval |
| `.tool(impl Tool)` | Attach a callable tool |
| `.tools(Vec<Box<dyn ToolDyn>>)` | Attach multiple tools |
| `.temperature(f64)` | Set temperature (0.0-1.0) |
| `.max_tokens(u64)` | Set max output tokens |
| `.additional_params(json!{...})` | Provider-specific params |
| `.tool_choice(ToolChoice)` | Control tool usage |
| `.build()` | Build the agent |

## Available Providers

Create a client with `ProviderName::Client::from_env()` or `ProviderName::Client::new("key")`.

| Provider | Module | Example Model Constant |
|----------|--------|----------------------|
| OpenAI | `openai` | `GPT_4O`, `GPT_4O_MINI` |
| Anthropic | `anthropic` | `CLAUDE_4_OPUS`, `CLAUDE_4_SONNET` |
| Cohere | `cohere` | `COMMAND_R_PLUS` |
| Mistral | `mistral` | `MISTRAL_LARGE` |
| Gemini | `gemini` | model string |
| Groq | `groq` | model string |
| Ollama | `ollama` | model string |
| DeepSeek | `deepseek` | model string |
| xAI | `xai` | model string |
| Together | `together` | model string |
| Perplexity | `perplexity` | model string |
| OpenRouter | `openrouter` | model string |
| HuggingFace | `huggingface` | model string |
| Azure | `azure` | deployment string |
| Hyperbolic | `hyperbolic` | model string |
| Galadriel | `galadriel` | model string |
| Moonshot | `moonshot` | model string |
| Mira | `mira` | model string |
| Voyage AI | `voyageai` | embeddings only |

## Vector Store Crates

| Backend | Crate |
|---------|-------|
| In-memory | `rig-core` (built-in) |
| MongoDB | `rig-mongodb` |
| LanceDB | `rig-lancedb` |
| Qdrant | `rig-qdrant` |
| SQLite | `rig-sqlite` |
| Neo4j | `rig-neo4j` |
| Milvus | `rig-milvus` |
| SurrealDB | `rig-surrealdb` |

## Key Rules

- All async code runs on tokio.
- Use `WasmCompatSend` / `WasmCompatSync` instead of raw `Send` / `Sync` for WASM compatibility.
- Use proper error types with `thiserror` — never `Result<(), String>`.
- Avoid `.unwrap()` — use `?` operator.

## Further Reference

Detailed API documentation (available when installed via Claude Code skills):
- **tools** — Tool trait, ToolDefinition, ToolEmbedding, attachment patterns
- **rag** — Vector stores, Embed derive, EmbeddingsBuilder, search requests
- **providers** — Provider-specific initialization, model constants, env vars
- **patterns** — Multi-agent, hooks, streaming details, chaining, extraction

For the full reference, see the Rig examples at `rig-core/examples/` or https://docs.rig.rs
