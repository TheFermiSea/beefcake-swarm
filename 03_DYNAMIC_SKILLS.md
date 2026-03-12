# Phase 3: Declarative Skills & Dynamic Tools

## Inspiration Context
OpenAI's Symphony utilizes `.codex/skills/` directories, each containing a `SKILL.md` (instructions) and optional scripts (`land_watch.py`). This allows hot-swapping agent capabilities without recompiling the harness.

## Agent Instructions

### 1. Define the Skill Manifest
Create `coordination/src/registry/skill_manifest.rs`.

```rust
#[derive(Deserialize, Debug)]
pub struct SkillManifest {
    pub name: String,
    pub description: String,
    pub system_prompt_file: String, // Path to SKILL.md
    pub tools: Vec<String>, // Names of Rust Rig-tools to map
    pub required_isolation: IsolationLevel,
}
```

### 2. Build the Dynamic Tool Injector
In `crates/swarm-agents/src/tools/bundles.rs`, create a `SkillLoader`.

- On startup, `beefcake-swarm` should scan `.claude/skills/` or a new `.beefcake/skills/` directory.
- For each directory, parse `manifest.json` and `SKILL.md`.
- When Rig builds an agent, instead of hardcoding `builder.tool(FileExistsTool)`, map the strings in `manifest.tools` to the actual `rig::Tool` implementations using an enum dispatcher (see Rig's `enum_dispatch.rs` example).

### 3. Contextual Injection
When configuring the `AgentBuilder`, append the contents of `SKILL.md` to the agent's system prompt dynamically.

```rust
// Pseudo-code
for skill in active_skills {
    let skill_prompt = fs::read_to_string(&skill.system_prompt_file)?;
    agent_builder = agent_builder.preamble(&format!("Skill {}:\n{}", skill.name, skill_prompt));
    for tool_name in &skill.tools {
        let rig_tool = tool_registry.get(tool_name)?;
        agent_builder = agent_builder.dynamic_tool(rig_tool); // Using Rig's tool injection
    }
}
```
