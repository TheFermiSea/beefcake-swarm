---
name: cloud-proxy
description: Manage the CLIAPIProxy (Router-for-Me) on ai-proxy — check status, list models, view credentials, re-auth stale tokens, restart, and troubleshoot. Use when the user asks about cloud proxy health, model availability, credential issues, or swarm cloud connectivity.
---

# CLIAPIProxy Management

The swarm's cloud tier runs through CLIAPIProxy (Router-for-Me) on ai-proxy at `http://100.105.113.58:8317`.

## Architecture

CLIAPIProxy is an OpenAI-compatible reverse proxy that multiplexes across multiple upstream providers (Anthropic, Google AI Studio, OpenAI/Codex) using OAuth tokens. It provides:
- Unified `/v1/chat/completions` endpoint for all providers
- Round-robin credential rotation with automatic quota failover
- Auto-refresh of OAuth tokens every 15 minutes
- 4-model fallback cascade for the swarm

## Quick Diagnostics

Run these from the local Mac (no SSH needed):

```bash
# Health check — list models (inference API)
curl -s -H "x-api-key: rust-daq-proxy-key" \
  http://100.105.113.58:8317/v1/models | python3 -m json.tool

# Test inference
curl -s -H "x-api-key: rust-daq-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-6","max_tokens":10,"messages":[{"role":"user","content":"Say OK"}]}' \
  http://100.105.113.58:8317/v1/chat/completions

# Credential health (management API)
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  http://100.105.113.58:8317/v0/management/auth-files | python3 -m json.tool

# Usage statistics
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  http://100.105.113.58:8317/v0/management/usage | python3 -m json.tool

# Stream recent logs
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  http://100.105.113.58:8317/v0/management/logs

# Download config
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  http://100.105.113.58:8317/v0/management/config.yaml
```

## Authentication

Two separate auth planes, same key value:

| Plane | Header | Base Path |
|-------|--------|-----------|
| Inference | `x-api-key: rust-daq-proxy-key` | `/v1/` |
| Management | `Authorization: Bearer rust-daq-proxy-key` | `/v0/management/` |

## Swarm Fallback Cascade

All 4 models must be present in `/v1/models` for the swarm to operate correctly:

| Priority | Model | Expected `owned_by` |
|----------|-------|---------------------|
| 1 (primary) | `claude-opus-4-6` | `anthropic` |
| 2 | `gemini-3.1-pro-high` | `antigravity` |
| 3 | `claude-sonnet-4-6` | `antigravity` |
| 4 | `gemini-3.1-flash-lite-preview` | `google` |

To test all four:
```bash
for model in claude-opus-4-6 gemini-3.1-pro-high claude-sonnet-4-6 gemini-3.1-flash-lite-preview; do
  echo -n "$model: "
  curl -s -w "%{http_code}" -o /dev/null \
    -H "x-api-key: rust-daq-proxy-key" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}" \
    http://100.105.113.58:8317/v1/chat/completions
  echo
done
```

## Credential Types

Credentials are stored in `/root/.cli-proxy-api/` on ai-proxy:

| Type | File Pattern | Auth Method | Auto-Refresh |
|------|-------------|-------------|-------------|
| Google AI Studio | `antigravity-*.json` | OAuth (refresh_token) | Every 15 min |
| Anthropic | `claude-*.json` | OAuth (refresh_token) | Every 15 min |
| OpenAI/Codex | `codex-*.json` | OAuth (refresh_token) | Every 15 min |
| GitHub Copilot | `github-copilot-*.json` | Static PAT | No |
| Vertex (legacy) | `*-gen-lang-client-*.json` | Gemini CLI OAuth | No (stale) |

### Detecting Stale Credentials

Check the `modtime` field in the auth-files response. If a credential's modtime is >24h old while others of the same type are fresh, its refresh_token may be revoked.

```bash
# Quick stale check
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  http://100.105.113.58:8317/v0/management/auth-files | \
  python3 -c "
import sys, json
from datetime import datetime, timezone
data = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for f in data.get('files', []):
    mod = datetime.fromisoformat(f['modtime'].replace('Z', '+00:00'))
    age_h = (now - mod).total_seconds() / 3600
    status = 'STALE' if age_h > 24 else 'ok'
    print(f'{status:5s} {age_h:6.1f}h  {f[\"name\"]}')"
```

## Re-Authentication

### Google AI Studio (antigravity) — OAuth re-auth

From ai-proxy:
```bash
ssh root@100.105.113.58
/opt/cli-proxy-api/cli-proxy-api --login --project_id <PROJECT_ID>
# Follow browser OAuth flow
```

Or initiate remotely via management API:
```bash
curl -s -H "Authorization: Bearer rust-daq-proxy-key" \
  "http://100.105.113.58:8317/v0/management/gemini-cli-auth-url?project_id=<PROJECT_ID>"
# Returns a URL to open in browser; requires SSH tunnel for callback
```

Known project IDs:
- `round-parity-3qhgw` (squires.b@gmail.com)
- `sigma-informatics-jns1n` (neogilabunt@gmail.com)
- `formidable-rune-xbc5s` (easternanemone@gmail.com)

### Anthropic (Claude) — OAuth re-auth

```bash
ssh root@100.105.113.58
/opt/cli-proxy-api/cli-proxy-api --anthropic-login
```

### OpenAI (Codex) — OAuth re-auth

```bash
ssh root@100.105.113.58
/opt/cli-proxy-api/cli-proxy-api --codex-login
```

## Process Management

The proxy runs as a bare process (no systemd). To restart:

```bash
# Find PID
ssh root@100.105.113.58 'pgrep -f cli-proxy-api'

# Restart
ssh root@100.105.113.58 'pkill -f cli-proxy-api; sleep 2; nohup /opt/cli-proxy-api/cli-proxy-api -config /opt/cli-proxy-api/config.yaml > /tmp/cliproxyapi.log 2>&1 &'

# Verify
curl -s -H "x-api-key: rust-daq-proxy-key" http://100.105.113.58:8317/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"data\"])} models')"
```

## Config Location

- Binary: `/opt/cli-proxy-api/cli-proxy-api`
- Config: `/opt/cli-proxy-api/config.yaml`
- Auth dir: `/root/.cli-proxy-api/`
- Logs: `/root/.cli-proxy-api/logs/main.log`
- Version: 6.8.54 (2026-03-15)

## Management API Reference

Full docs: https://help.router-for.me/management/api

Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v0/management/config` | Full config as JSON |
| GET/PUT | `/v0/management/config.yaml` | Download/replace YAML config |
| GET | `/v0/management/auth-files` | List credentials with status |
| DELETE | `/v0/management/auth-files?name=<file>` | Remove a credential |
| POST | `/v0/management/auth-files` | Upload a credential |
| GET | `/v0/management/usage` | Usage statistics |
| GET/PUT | `/v0/management/debug` | Toggle debug mode |
| GET | `/v0/management/logs` | Stream recent logs |
| GET/PATCH | `/v0/management/api-keys` | Manage proxy auth keys |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `{"error":"Missing API key"}` | No `x-api-key` header | Add `-H "x-api-key: rust-daq-proxy-key"` |
| Model returns 403/429 | Credential quota exhausted | Proxy auto-retries with next credential; check auth-files for disabled ones |
| Model not in `/v1/models` | Credential for that provider expired | Re-auth the relevant credential type |
| Connection refused on 8317 | Process died | Restart (see Process Management above) |
| Stale credential (modtime >24h) | Refresh token revoked | Re-authenticate that account |
