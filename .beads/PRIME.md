# BeadHub Workspace

> Use `bd` for all issue operations. Use `bdh` only for coordination commands (`:status`, `:policy`, `:run`, `:aweb`).

**Start every session:**
```bash
bdh :status    # your identity + team status (bdh-only command)
bdh :policy    # READ AND FOLLOW (bdh-only command)
bd ready       # find work
```

**Before ending session:**
```bash
git status
git add <files>
bd dolt commit
git commit -m "..."
bd dolt push
git push
```

---

# Beads Workflow Context

> **Context Recovery**: Run `bd prime` after compaction, clear, or new session
> Hooks auto-call this in Claude Code when .beads/ detected

# 🚨 SESSION CLOSE PROTOCOL 🚨

**CRITICAL**: Before saying "done" or "complete", you MUST run this checklist:

```
[ ] 1. git status              (check what changed)
[ ] 2. git add <files>         (stage code changes)
[ ] 3. git commit -m "..."     (commit code)
[ ] 4. git push                (push to remote)
```

**NEVER skip this.** Work is not done until pushed.

## Core Rules
- **Default**: Use beads for ALL task tracking (`bd create`, `bd ready`, `bd close`)
- **Prohibited**: Do NOT use TodoWrite, TaskCreate, or markdown files for task tracking
- **Workflow**: Create beads issue BEFORE writing code, mark in_progress when starting
- **Memory**: Use `bd remember "insight"` for persistent knowledge across sessions. Do NOT use MEMORY.md files — they fragment across accounts. Search with `bd memories <keyword>`.
- Persistence you don't need beats lost context
- Git workflow: beads auto-commit to Dolt, run `git push` at session end
- Session management: check `bd ready` for available work
- **`bd` vs `bdh`**: `bd` (v0.62+) has ALL issue tracking + memory features. `bdh` (v0.11+) adds coordination commands prefixed with `:` (`:status`, `:policy`, `:run`, `:aweb`, `:escalate`). Use `bd` by default; only use `bdh` for `:` commands.

## Essential Commands

### Finding Work
- `bd ready` - Show issues ready to work (no blockers)
- `bd list --status=open` - All open issues
- `bd list --status=in_progress` - Your active work
- `bd show <id>` - Detailed issue view with dependencies

### Creating & Updating
- `bd create --title="Summary of this issue" --description="Why this issue exists and what needs to be done" --type=task|bug|feature --priority=2` - New issue
  - Priority: 0-4 or P0-P4 (0=critical, 2=medium, 4=backlog). NOT "high"/"medium"/"low"
- `bd update <id> --status=in_progress` - Claim work
- `bd update <id> --assignee=username` - Assign to someone
- `bd update <id> --title/--description/--notes/--design` - Update fields inline
- `bd close <id>` - Mark complete
- `bd close <id1> <id2> ...` - Close multiple issues at once (more efficient)
- `bd close <id> --reason="explanation"` - Close with reason
- **Tip**: When creating multiple issues/tasks/epics, use parallel subagents for efficiency
- **WARNING**: Do NOT use `bd edit` - it opens $EDITOR (vim/nano) which blocks agents

### Dependencies & Blocking
- `bd dep add <issue> <depends-on>` - Add dependency (issue depends on depends-on)
- `bd blocked` - Show all blocked issues
- `bd show <id>` - See what's blocking/blocked by this issue

### Sync & Collaboration
- `bd dolt push` - Push beads to Dolt remote
- `bd dolt pull` - Pull beads from Dolt remote
- `bd search <query>` - Search issues by keyword

### Project Health
- `bd stats` - Project statistics (open/closed/blocked counts)
- `bd doctor` - Check for issues (sync problems, missing hooks)

### Coordination (bdh-only commands)
- `bdh :status` - Show identity + team status
- `bdh :policy` - Show project policy and role playbook
- `bdh :run` - Run an AI coding agent in a loop
- `bdh :aweb` - Agent messaging (mail/chat/locks)
- `bdh :escalate` - Escalate to human when stuck
- `bdh :reservations` - File locking to prevent conflicts

## Common Workflows

**Starting work:**
```bash
bd ready           # Find available work
bd show <id>       # Review issue details
bd update <id> --status=in_progress  # Claim it
```

**Completing work:**
```bash
bd close <id1> <id2> ...    # Close all completed issues at once
git add . && git commit -m "..."  # Commit code changes
git push                    # Push to remote
```

**Creating dependent work:**
```bash
bd create --title="Implement feature X" --description="Why this issue exists and what needs to be done" --type=feature
bd create --title="Write tests for X" --description="Why this issue exists and what needs to be done" --type=task
bd dep add beads-yyy beads-xxx  # Tests depend on Feature (Feature blocks tests)
```

## Persistent Memories

Stored via `bd remember`. Search with `bd memories <keyword>`. Remove with `bd forget <key>`.

### beadhub-deployed-on-ai-proxy-ct-800-at
BeadHub deployed on ai-proxy (CT 800) at port 8080. Docker stack: PostgreSQL 16 + Redis 7 + FastAPI API. Port 8000 blocked by SurrealDB (pid=229). Two workspaces registered: brian-mac (local macOS, lead) and swarm-lead (ai-proxy, lead). bdh v0.10.4 installed on both. SWARM_BEADS_BIN defaults to bdh in run-swarm.sh. BeadHub URL: http://localhost:8080 from ai-proxy, http://100.105.113.58:8080 from external. Start: cd /home/brian/code/beadhub && POSTGRES_PASSWORD=beadhub-local-dev BEADHUB_PORT=8080 docker compose up -d. Issue beefcake-z03n tracks wiring bdh :init into worktree creation.

### dolt-remotesapi-configured-2026-03-09-ai-proxy
Dolt shared-server mode is authoritative for beefcake-swarm. The live Dolt sql-server data/config live under `~/.beads/shared-server/dolt`, not `~/code/beefcake-swarm/.beads/dolt`. On the local Mac, `bd context` is authoritative for the active server binding; `bd where`, `bd info`, and `--db` can still show direct-path values for compatibility in bd v0.62.0. Local Mac remote: `origin → http://100.105.113.58:8001/beads`. ai-proxy remote: `origin → http://localhost:8001/beads`. Start/restart the local shared server from the repo root with `bd dolt start` / `bd dolt stop`. Use `scripts/bd-safe.sh doctor|context|where|info|federation list-peers` for admin diagnostics, because `bd federation list-peers` can surface the Dolt `origin` remote even when `select * from federation_peers` returns no rows. Do **not** run `bd federation remove-peer origin`; that removes the Dolt replication remote used by `bd dolt push/pull`.

### file-targeting-fix-find-target-files-by-grep
file-targeting-fix: find_target_files_by_grep now extracts snake_case identifiers (e.g., edit_file) in addition to CamelCase. Also added path-based scoring boosts: tools/ +2, patches/ -2, tests/ -1. dogfood-loop.sh now passes title+description (first 300 chars) as the objective. Commits 729e36c and 76c69b0.

### parallel-dispatch
Parallel dispatch + round-robin routing shipped in PR #45 (feat/parallel-dispatch-round-robin). Key: process_issue is !Send (dyn tracing::Value: !Sync), so parallel dispatch uses std::thread::spawn + Handle::block_on, NOT JoinSet. Cooperative cancellation via Arc<AtomicBool> cancel param on process_issue.
