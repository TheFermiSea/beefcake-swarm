#!/bin/bash
# Apply a TZ experiment profile to tensorzero.toml and restart the gateway.
#
# Reads candidate_variants from config/experiments/<profile>.toml and patches
# the corresponding [functions.*.experimentation] sections in tensorzero.toml.
#
# Usage:
#   ./scripts/tz-apply-experiment.sh normal           # restore defaults
#   ./scripts/tz-apply-experiment.sh gemma-experiment  # activate gemma
#   ./scripts/tz-apply-experiment.sh --current         # show active profile
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TZ_CONFIG="$REPO_ROOT/config/tensorzero.toml"
PROFILES_DIR="$REPO_ROOT/config/experiments"
ACTIVE_FILE="$PROFILES_DIR/.active"

show_current() {
    if [ -f "$ACTIVE_FILE" ]; then
        echo "Active profile: $(cat "$ACTIVE_FILE")"
    else
        echo "Active profile: unknown (no .active file)"
    fi
    echo ""
    echo "Current candidate_variants:"
    grep 'candidate_variants = \[' "$TZ_CONFIG" | head -6
}

apply_profile() {
    local profile="$1"
    local profile_file="$PROFILES_DIR/${profile}.toml"

    if [ ! -f "$profile_file" ]; then
        echo "ERROR: Profile not found: $profile_file" >&2
        echo "Available profiles:"
        ls "$PROFILES_DIR"/*.toml 2>/dev/null | xargs -I{} basename {} .toml
        exit 1
    fi

    echo "Applying experiment profile: $profile"

    python3 - "$TZ_CONFIG" "$profile_file" <<'PYEOF'
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

config_path, profile_path = sys.argv[1], sys.argv[2]

# Read profile
with open(profile_path, 'rb') as f:
    profile = tomllib.load(f)

# Read config as text (we'll do targeted replacement to preserve formatting)
with open(config_path) as f:
    lines = f.readlines()

# For each function in the profile, find its experimentation section and
# replace the candidate_variants line
changes = 0
for func_name, func_profile in profile.items():
    if 'candidate_variants' not in func_profile:
        continue

    new_candidates = func_profile['candidate_variants']
    # TOML requires double quotes — Python str(list) uses single quotes
    import json
    new_line = f'candidate_variants = {json.dumps(new_candidates)}\n'

    # Find the [functions.<func_name>.experimentation] section
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f'[functions.{func_name}.experimentation]':
            in_section = True
            continue
        if in_section and stripped.startswith('candidate_variants'):
            old = lines[i].strip()
            lines[i] = new_line
            changes += 1
            print(f"  {func_name}: {old}")
            print(f"         → {new_line.strip()}")
            in_section = False
            continue
        # Exit section if we hit a new section header
        if in_section and stripped.startswith('['):
            in_section = False

if changes == 0:
    print("WARNING: No candidate_variants lines were updated!")
    sys.exit(1)

with open(config_path, 'w') as f:
    f.writelines(lines)

print(f"\nUpdated {changes} experiment section(s).")
PYEOF

    # Record active profile
    echo "$profile" > "$ACTIVE_FILE"

    # Restart TZ gateway
    echo "Restarting TZ gateway..."
    docker restart tensorzero-gateway-1 >/dev/null 2>&1
    sleep 3

    # Verify
    if docker logs --since 5s tensorzero-gateway-1 2>&1 | grep -q "listening"; then
        echo "TZ gateway healthy with profile: $profile"
    else
        echo "WARNING: TZ gateway may have failed to start. Check: docker logs tensorzero-gateway-1"
    fi
}

case "${1:-}" in
    --current|-c) show_current ;;
    "") echo "Usage: $0 <profile-name> | --current"; echo "Profiles:"; ls "$PROFILES_DIR"/*.toml 2>/dev/null | xargs -I{} basename {} .toml; exit 1 ;;
    *) apply_profile "$1" ;;
esac
