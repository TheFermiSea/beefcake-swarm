# BeadHub Workspace

> Always use `bdh` (not `bd`) — it coordinates work across agents.

**Start every session:**
```bash
bdh :status    # your identity + team status
bdh :policy        # READ AND FOLLOW
bdh ready          # find work
```

**Before ending session:**
```bash
git status
git add <files>
bdh sync
git commit -m "..."
bdh sync
git push
```

---

# Beads Workflow Context

> **Context Recovery**: Run `bdh prime` after compaction, clear, or new session
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
- **Default**: Use beads for ALL task tracking (`bdh create`, `bdh ready`, `bdh close`)
- **Prohibited**: Do NOT use TodoWrite, TaskCreate, or markdown files for task tracking
- **Workflow**: Create beads issue BEFORE writing code, mark in_progress when starting
- **Memory**: Use `bdh remember "insight"` for persistent knowledge across sessions. Do NOT use MEMORY.md files — they fragment across accounts. Search with `bdh memories <keyword>`.
- Persistence you don't need beats lost context
- Git workflow: beads auto-commit to Dolt, run `git push` at session end
- Session management: check `bdh ready` for available work

## Essential Commands

### Finding Work
- `bdh ready` - Show issues ready to work (no blockers)
- `bdh list --status=open` - All open issues
- `bdh list --status=in_progress` - Your active work
- `bdh show <id>` - Detailed issue view with dependencies

### Creating & Updating
- `bdh create --title="Summary of this issue" --description="Why this issue exists and what needs to be done" --type=task|bug|feature --priority=2` - New issue
  - Priority: 0-4 or P0-P4 (0=critical, 2=medium, 4=backlog). NOT "high"/"medium"/"low"
- `bdh update <id> --status=in_progress` - Claim work
- `bdh update <id> --assignee=username` - Assign to someone
- `bdh update <id> --title/--description/--notes/--design` - Update fields inline
- `bdh close <id>` - Mark complete
- `bdh close <id1> <id2> ...` - Close multiple issues at once (more efficient)
- `bdh close <id> --reason="explanation"` - Close with reason
- **Tip**: When creating multiple issues/tasks/epics, use parallel subagents for efficiency
- **WARNING**: Do NOT use `bdh edit` - it opens $EDITOR (vim/nano) which blocks agents

### Dependencies & Blocking
- `bdh dep add <issue> <depends-on>` - Add dependency (issue depends on depends-on)
- `bdh blocked` - Show all blocked issues
- `bdh show <id>` - See what's blocking/blocked by this issue

### Sync & Collaboration
- `bdh dolt push` - Push beads to Dolt remote
- `bdh dolt pull` - Pull beads from Dolt remote
- `bdh search <query>` - Search issues by keyword

### Project Health
- `bdh stats` - Project statistics (open/closed/blocked counts)
- `bdh doctor` - Check for issues (sync problems, missing hooks)

## Common Workflows

**Starting work:**
```bash
bdh ready           # Find available work
bdh show <id>       # Review issue details
bdh update <id> --status=in_progress  # Claim it
```

**Completing work:**
```bash
bdh close <id1> <id2> ...    # Close all completed issues at once
git add . && git commit -m "..."  # Commit code changes
git push                    # Push to remote
```

**Creating dependent work:**
```bash
# Run bdh create commands in parallel (use subagents for many items)
bdh create --title="Implement feature X" --description="Why this issue exists and what needs to be done" --type=feature
bdh create --title="Write tests for X" --description="Why this issue exists and what needs to be done" --type=task
bdh dep add beads-yyy beads-xxx  # Tests depend on Feature (Feature blocks tests)
```

## Persistent Memories (4)

Stored via `bdh remember`. Search with `bdh memories <keyword>`. Remove with `bdh forget <key>`.

### beadhub-deployed-on-ai-proxy-ct-800-at
BeadHub deployed on ai-proxy (CT 800) at port 8080. Docker stack: PostgreSQL 16 + Redis 7 + FastAPI API. Port 8000 blocked by SurrealDB (pid=229). Two workspaces registered: brian-mac (local macOS, lead) and swarm-lead (ai-proxy, lead). bdh v0.10.4 installed on both. SWARM_BEADS_BIN defaults to bdh in run-swarm.sh. BeadHub URL: http://localhost:8080 from ai-proxy, http://100.105.113.58:8080 from external. Start: cd /home/brian/code/beadhub && POSTGRES_PASSWORD=beadhub-local-dev BEADHUB_PORT=8080 docker compose up -d. Issue beefcake-z03n tracks wiring bdh :init into worktree creation.

### dolt-remotesapi-configured-2026-03-09-ai-proxy
Dolt remotesapi configured 2026-03-09. ai-proxy runs dolt sql-server: SQL on port 3307 (0.0.0.0), remotesapi on port 8001 (0.0.0.0). Config: ~/code/beefcake-swarm/.beads/dolt/config.yaml. Privileges: root@% has ALL+CLONE_ADMIN+SUPER. Local Mac remote: origin → http://100.105.113.58:8001/beads. ai-proxy remote: origin → http://localhost:8001/beads. Start: cd ~/code/beefcake-swarm/.beads/dolt && nohup dolt sql-server --config config.yaml > /tmp/dolt-remotesapi.log 2>&1 &. Always bd dolt commit before pull (fails with uncommitted changes). Data in beads database (550 issues).

### file-targeting-fix-find-target-files-by-grep
file-targeting-fix: find_target_files_by_grep now extracts snake_case identifiers (e.g., edit_file) in addition to CamelCase. Also added path-based scoring boosts: tools/ +2, patches/ -2, tests/ -1. dogfood-loop.sh now passes title+description (first 300 chars) as the objective. Commits 729e36c and 76c69b0.

### parallel-dispatch
Parallel dispatch + round-robin routing shipped in PR #45 (feat/parallel-dispatch-round-robin). Key: process_issue is !Send (dyn tracing::Value: !Sync), so parallel dispatch uses std::thread::spawn + Handle::block_on, NOT JoinSet. Cooperative cancellation via Arc<AtomicBool> cancel param on process_issue.

