# Phase 2: Ephemeral Workspace Isolation

## Inspiration Context
Symphony uses `workspace.ex` and `path_safety.ex` to ensure agents don't corrupt the main repository. Work is done in an isolated environment and synced back only upon successful verification.

## Agent Instructions

### 1. Create the `Workspace` Struct
Create `crates/swarm-agents/src/workspace_manager.rs`.
Define a `Workspace` struct that represents an ephemeral environment.

```rust
pub struct Workspace {
    pub id: String,
    pub base_path: PathBuf,
    pub worktree_path: PathBuf,
    // Determines if this workspace uses a git worktree, a local temp dir, or a Docker container
    pub isolation_level: IsolationLevel,
}
```

### 2. Implement Git Worktree Strategy
When the Orchestrator initializes a task:

- Generate a unique ID (e.g., UUID or short hash).
- Execute `git worktree add ../.beefcake-workspaces/<id> <base_branch>`.
- Set the `worktree_path` as the root directory for ALL tools provided to the Rig agent in this specific workflow state.

### 3. Enforce Path Safety
Implement a middleware in `coordination/src/shell_safety.rs` and `crates/swarm-agents/src/tools/fs_tools.rs`.

- Rule: Every File System read/write tool MUST sanitize the input path. If `canonicalize(path)` resolves outside of `workspace.worktree_path`, return a `ToolError::SecurityViolation`.
- Rule: Shell execution tools (`exec_tool.rs`) must have their `current_dir` explicitly set to `workspace.worktree_path`.

### 4. Implement the 'Land' Skill
Create a capability to sync back to main (similar to Symphony's `land/SKILL.md`).

Once the Verification state passes and reaches `HandoffReady`, the Orchestrator triggers `Workspace::commit_and_push()`, which merges the worktree back into the origin branch and cleans up the ephemeral worktree.
