# NotebookLM Integration — beefcake-swarm

**Version:** 2.0
**Status:** Implemented (Phases 2-6 complete, notebooks pending creation/seeding)

## Overview

The beefcake-swarm uses NotebookLM as an external RAG layer for institutional memory. This replaces the previously empty "Learning Layer" fields in `WorkPacket` (`relevant_heuristics`, `relevant_playbooks`, `decisions`) with live knowledge queries.

## Architecture

```
Orchestrator Loop
  |
  |--- Pre-task: query Project Brain + Debugging KB → populate WorkPacket fields
  |--- Format prompt: render knowledge fields in agent context
  |--- Pre-escalation: check Debugging KB before tier escalation
  |--- Post-success: capture resolution + error patterns
  |
  Manager Agent
  |--- query_notebook tool: on-demand knowledge queries
```

### Modules

| Module | File | Purpose |
|--------|------|---------|
| `NotebookBridge` | `crates/swarm-agents/src/notebook_bridge.rs` | CLI wrapper (`nlm`), `KnowledgeBase` trait, registry parsing |
| `QueryNotebookTool` | `crates/swarm-agents/src/tools/notebook_tool.rs` | Rig tool for Manager agents |
| `knowledge_sync` | `crates/swarm-agents/src/knowledge_sync.rs` | Automated capture: resolutions, error patterns, codebase sync |
| Registry | `notebook_registry.toml` | Role-to-notebook-ID mapping (TOML) |

### KnowledgeBase Trait

```rust
pub trait KnowledgeBase: Send + Sync {
    fn query(&self, role: &str, question: &str) -> Result<String>;
    fn add_source_text(&self, role: &str, title: &str, content: &str) -> Result<()>;
    fn add_source_file(&self, role: &str, file_path: &str) -> Result<()>;
    fn is_available(&self) -> bool;
}
```

Implementations: `NotebookBridge` (real), `NoOpKnowledgeBase` (fallback), `MockKnowledgeBase` (tests).

### Graceful Degradation

The knowledge base is fully optional. When `nlm` is unavailable or the registry is missing:
- `NotebookBridge::is_available()` returns `false`
- Orchestrator runs without knowledge queries (no errors)
- Manager agent has no `query_notebook` tool (gracefully absent)
- All `knowledge_sync` functions log warnings but don't fail the pipeline

## Setup

### 1. Install CLI
```bash
uv tool install notebooklm-mcp-cli
nlm login
```

### 2. Create Notebooks
```bash
nlm notebook create "beefcake-swarm: Project Brain"
nlm notebook create "beefcake-swarm: Debugging KB"
nlm notebook create "beefcake-swarm: Codebase"
nlm notebook create "beefcake-swarm: Research"
nlm notebook create "beefcake-swarm: Security"
nlm notebook create "beefcake-swarm: Visuals"
```

### 3. Update Registry
Edit `notebook_registry.toml` and fill in the notebook IDs from step 2.

### 4. Seed Notebooks
```bash
# Project Brain
nlm source add "<BRAIN_ID>" --file CLAUDE.md

# Codebase
repomix --style markdown --output /tmp/beefcake-swarm-repomix.md
nlm source add "<CODEBASE_ID>" --file /tmp/beefcake-swarm-repomix.md
```

## Complementary Tools

| Tool | Scope |
|------|-------|
| CocoIndex | Code structure — callers, implementors, file navigation |
| NotebookLM | Knowledge — decisions, patterns, docs, error playbooks |
| Beads | Issue tracking — what needs to be done |
| Repomix | Feeds NotebookLM with packed codebase context |

## Environment Variables

- `SWARM_NLM_BIN` — Override the `nlm` binary path (default: `"nlm"`)

## CLI Commands

| Action | Command |
|--------|---------|
| Login | `nlm login` |
| List Notebooks | `nlm notebook list` |
| Create Notebook | `nlm notebook create "Title"` |
| Add File Source | `nlm source add "ID" --file "doc.txt"` |
| Add Text Source | `nlm source add "ID" --text "content" --title "title"` |
| Query (RAG) | `nlm query notebook "ID" "Question..."` |

## Testing

```bash
# Unit tests (includes NotebookBridge, knowledge_sync, mock KB)
cargo test -p swarm-agents

# Manual verification
nlm --help                                    # CLI available
nlm notebook list                             # Auth working
nlm query notebook "<ID>" "test query"      # Query working
```
