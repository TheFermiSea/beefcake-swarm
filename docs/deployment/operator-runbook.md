# Beefcake Swarm — Operator Runbook

**Audience:** On-call operators and developers  
**Scope:** Restart order, endpoint probes, and swarm resume strategy  
**Last updated:** 2026-02-27

---

## 1. Restart Order

Bring components up in this order. Each step depends on the previous.

```
1. NFS server (slurm-ctl)
2. Inference endpoints (vasp-02, then vasp-01)
3. Cloud proxy / ai-proxy
4. SLURM swarm jobs
```

### 1.1 NFS server

```bash
ssh root@10.0.0.5
systemctl status nfs-kernel-server   # check health
# If down:
systemctl restart nfs-kernel-server
exportfs -rav                         # re-export /cluster/shared
```

Verify mounts from any compute node:

```bash
ssh root@10.0.0.21 "df -h /cluster/shared"
```

### 1.2 Inference endpoints

**vasp-02 (HydraCoder 30B — all tiers)**

```bash
ssh root@10.0.0.21
# Check if already running
pgrep -a llama-server
# Start HydraCoder (manual, no SLURM while NFS is unavailable)
nohup /tmp/start-hydracoder.sh > /tmp/hydracoder-server.log 2>&1 &
```

Verify endpoint health:

```bash
curl -fsS http://10.0.0.21:8080/health && echo " vasp-02 OK"
```

**vasp-01 (Qwen3.5-397B — deferred until disk space restored)**

```bash
ssh root@10.0.0.20
pgrep -a llama-server
# When Qwen3.5 download completes:
nohup /tmp/start-qwen35.sh > /tmp/qwen35-server.log 2>&1 &
```

Verify:

```bash
curl -fsS http://10.0.0.20:8081/health && echo " vasp-01 OK"
```

### 1.3 Cloud proxy (ai-proxy)

The CLIAPIProxy routes cloud model requests to Anthropic/OpenAI.

```bash
ssh brian@100.105.113.58
systemctl --user status cloud-proxy
# If down:
systemctl --user restart cloud-proxy
# Verify:
curl -fsS http://10.0.0.5:8317/health && echo " proxy OK"
curl http://10.0.0.5:8317/v1/models | python3 -m json.tool | grep '"id"' | head -5
```

### 1.4 SLURM swarm jobs

```bash
ssh root@10.0.0.5
# Submit a single issue:
sbatch /cluster/shared/code/beefcake-swarm/scripts/run-swarm-sandbox.slurm \
  --export=ALL,SWARM_ISSUE_ID=beefcake-XXXX

# Submit a batch (sequential):
cd /cluster/shared/code/beefcake-swarm
./scripts/submit-swarm-batch.sh

# Watch queue:
watch -n 5 squeue -u brian
```

---

## 2. Endpoint Probes

### 2.1 Quick health check (all endpoints)

```bash
#!/usr/bin/env bash
# Run from any node on the 10.0.0.x network
check() {
    local name=$1 url=$2
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
        echo "✓ $name ($url)"
    else
        echo "✗ $name UNREACHABLE ($url)"
    fi
}

check "vasp-02 HydraCoder"  http://10.0.0.21:8080/health
check "vasp-01 Qwen3.5"     http://10.0.0.20:8081/health
check "ai-proxy"            http://10.0.0.5:8317/health
check "NFS"                 "$(df /cluster/shared 2>&1 | grep -q cluster && echo OK || echo FAIL)"
```

### 2.2 Model smoke test

Verify a model actually generates tokens (not just health ping):

```bash
curl -sS http://10.0.0.21:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"HydraCoder.i1-Q4_K_M","messages":[{"role":"user","content":"Reply with one word: Ready"}],"max_tokens":5}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Expected output: `Ready` (or any single-word acknowledgement).

### 2.3 Endpoint discovery via NFS JSON

The SLURM job writes a discovery file on startup:

```bash
# From slurm-ctl or any NFS-mounted node:
cat /cluster/shared/ai/endpoints/<JOBID>-fast.json 2>/dev/null || echo "no endpoint file"
# Fields: host, port, model, started_at, health_url
```

---

## 3. Resume Strategy

### 3.1 Resuming a partially-completed issue

If a SLURM job died mid-iteration, the issue stays `in_progress` in beads.

```bash
# 1. Check the swarm resume file (written by orchestrator on each iteration):
cat /cluster/shared/wt/beefcake-XXXX/.swarm-resume.json 2>/dev/null | python3 -m json.tool

# 2. Check telemetry for last known state:
tail -20 /cluster/shared/code/beefcake-swarm/.swarm-telemetry.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        if d.get('issue_id') == 'beefcake-XXXX':
            print(d.get('phase'), d.get('iteration'), d.get('outcome'))
    except json.JSONDecodeError:
        pass
"

# 3. Resubmit. The resume file is picked up automatically:
sbatch /cluster/shared/code/beefcake-swarm/scripts/run-swarm-sandbox.slurm \
  --export=ALL,SWARM_ISSUE_ID=beefcake-XXXX
```

### 3.2 Stuck issue (loop with no file changes)

Symptom: repeated log lines `No file changes after agent response`.

```bash
# 1. Verify the agent is actually making tool calls:
grep -i "tool_call\|write_file\|edit_file" \
  /cluster/shared/ai/logs/swarm-orch-<JOBID>.log | tail -10

# 2. If no tool calls: the model is in text-analysis mode.
#    Force temperature bump via env var and resubmit:
sbatch /cluster/shared/code/beefcake-swarm/scripts/run-swarm-sandbox.slurm \
  --export=ALL,SWARM_ISSUE_ID=beefcake-XXXX,SWARM_WORKER_TEMP=0.1

# 3. If tool calls present but no git diff: the worktree may be on the wrong branch.
#    Clean it and resubmit:
git -C /cluster/shared/code/beefcake-swarm \
  worktree remove --force /cluster/shared/wt/beefcake-XXXX 2>/dev/null || true
rm -rf /cluster/shared/wt/beefcake-XXXX*
git -C /cluster/shared/code/beefcake-swarm branch -D swarm/beefcake-XXXX 2>/dev/null || true
git -C /cluster/shared/code/beefcake-swarm worktree prune
# Then resubmit normally.
```

### 3.3 Context overflow (HTTP 500 from llama-server)

Symptom: `HTTP 500 Failed to parse input at pos NNNNN` in SLURM logs.

```bash
# 1. Confirm it's a context limit, not a server crash:
grep "HTTP 500\|context\|pos " /cluster/shared/ai/logs/swarm-orch-<JOBID>.log | tail -5

# 2. Reduce max_turns for this job:
sbatch /cluster/shared/code/beefcake-swarm/scripts/run-swarm-sandbox.slurm \
  --export=ALL,SWARM_ISSUE_ID=beefcake-XXXX,SWARM_WORKER_MAX_TURNS=4

# 3. If the model is restarting from a checkpoint with large context,
#    clear the resume file to start fresh:
rm /cluster/shared/wt/beefcake-XXXX/.swarm-resume.json 2>/dev/null || true
```

### 3.4 Escalation to cloud manager failed

Symptom: `Cloud preflight failed` or `quota/auth` in logs.

```bash
# 1. Check API key validity:
curl -sS -H "Authorization: Bearer $(cat /cluster/shared/ai/.cloud-api-key)" \
  http://10.0.0.5:8317/v1/models | python3 -m json.tool | grep '"id"' | head -3

# 2. If key expired, update it:
echo "new-api-key-here" > /cluster/shared/ai/.cloud-api-key
chmod 600 /cluster/shared/ai/.cloud-api-key

# 3. Force local manager mode as temporary workaround:
sbatch /cluster/shared/code/beefcake-swarm/scripts/run-swarm-sandbox.slurm \
  --export=ALL,SWARM_ISSUE_ID=beefcake-XXXX,SWARM_USE_CLOUD=0
```

### 3.5 Reset an issue back to open

If the swarm cannot make progress and human intervention is needed:

```bash
bd update beefcake-XXXX --status open
bd annotate beefcake-XXXX "Returned to open — swarm exhausted budget. Needs human review."
```

---

## 4. Common Log Locations

| Source | Path |
|--------|------|
| SLURM swarm job | `/cluster/shared/ai/logs/swarm-orch-<JOBID>.log` |
| HydraCoder server | `/tmp/hydracoder-server.log` (on vasp-02) |
| Qwen3.5 server | `/tmp/qwen35-server.log` (on vasp-01) |
| Cloud proxy | `journalctl --user -u cloud-proxy` (on ai-proxy) |
| Swarm telemetry | `$REPO/.swarm-telemetry.jsonl` (JSONL, append-only) |
| Per-issue metrics | `$WORKTREE/.swarm-metrics.json` |
| Endpoint discovery | `/cluster/shared/ai/endpoints/<JOBID>-{fast,reasoning}.json` |

---

## 5. Emergency Contacts

| Role | Action |
|------|--------|
| Inference down > 30 min | Check SLURM node status: `sinfo -l` |
| NFS unavailable | Restart NFS on slurm-ctl (§1.1) |
| Cloud quota exhausted | Set `SWARM_USE_CLOUD=0` (§3.4) |
| Orphaned worktrees | `git -C $REPO worktree prune` |
| Beads DB corruption | `bd sync` to reconcile with git |
