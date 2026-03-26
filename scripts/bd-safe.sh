#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

shared_server_enabled() {
    grep -Eq '^dolt\.shared-server:\s*true$' .beads/config.yaml 2>/dev/null
}

has_db_override() {
    local prev=""
    for arg in "$@"; do
        if [[ "$prev" == "--db" ]]; then
            return 0
        fi
        prev="$arg"
    done
    return 1
}

print_shared_server_note() {
    cat >&2 <<'EOF'
[bd-safe] Shared-server mode is enabled.
[bd-safe] `bd context` is authoritative for the live server/database binding.
[bd-safe] `bd where`, `bd info`, and `--db` still expose direct-path values in bd v0.62.0 for compatibility.
EOF
}

print_origin_remote_note() {
    cat >&2 <<'EOF'
[bd-safe] `origin` is the Dolt replication remote surfaced on federation commands.
[bd-safe] `bd federation list-peers` can show `origin` even when `select * from federation_peers` returns no rows.
[bd-safe] Do not use `bd federation remove-peer origin`; use `bd dolt remote list|add|remove` for replication remotes.
EOF
}

print_vc_commit_note() {
    cat >&2 <<'EOF'
[bd-safe] Shared-server memory/config writes can remain in WORKING after `bd vc commit` in bd v0.62.0.
[bd-safe] Verify with:
[bd-safe]   bd sql -q "select to_key, from_key, diff_type from dolt_diff_config where to_commit = 'WORKING' or from_commit = 'WORKING'"
EOF
}

if shared_server_enabled; then
    if has_db_override "$@"; then
        echo "[bd-safe] Warning: --db does not override the server-bound database reported by \`bd context\` in bd v0.62.0 shared-server mode." >&2
    fi

    if [[ $# -ge 3 && "$1" == "federation" && "$2" == "remove-peer" && "$3" == "origin" ]]; then
        print_origin_remote_note
        exit 2
    fi
fi

tmp_output="$(mktemp)"
status=0
if ! bd "$@" >"$tmp_output" 2>&1; then
    status=$?
fi
cat "$tmp_output"

if shared_server_enabled; then
    if [[ $# -ge 1 && ( "$1" == "context" || "$1" == "where" || "$1" == "info" ) ]]; then
        print_shared_server_note
    fi

    if [[ $# -ge 2 && "$1" == "federation" && "$2" == "list-peers" ]]; then
        print_origin_remote_note
    fi

    if [[ $# -ge 1 && "$1" == "doctor" ]]; then
        if grep -q 'Federation remotesapi' "$tmp_output" && grep -q 'No federation peers configured (only origin remote)' "$tmp_output"; then
            print_origin_remote_note
        fi
    fi

    if [[ $# -ge 2 && "$1" == "vc" && "$2" == "commit" ]]; then
        if bd sql -q "select to_key, from_key, diff_type from dolt_diff_config where to_commit = 'WORKING' or from_commit = 'WORKING'" 2>/dev/null | grep -q 'kv.memory\.'; then
            print_vc_commit_note
        fi
    fi
fi

rm -f "$tmp_output"
exit "$status"
