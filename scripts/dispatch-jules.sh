#!/bin/bash
# dispatch-jules.sh — Batch-create GitHub Issues with "jules" label from beads backlog.
#
# Picks simple issues (unused functions, TODOs, small bugs) from `bd ready`
# and creates corresponding GitHub Issues labeled "jules" to trigger the
# Jules AI agent workflow. Each issue consumes one Jules session.
#
# Usage:
#   bash scripts/dispatch-jules.sh              # dispatch up to 20 issues
#   bash scripts/dispatch-jules.sh --max 50     # dispatch up to 50
#   bash scripts/dispatch-jules.sh --dry-run    # preview without creating
#   bash scripts/dispatch-jules.sh --pattern "Unused function"  # filter
#
set -euo pipefail

MAX_DISPATCH=20
DRY_RUN=false
PATTERN="Unused function|TODO:|Large file:"
REPO="TheFermiSea/beefcake-swarm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_DISPATCH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Get already-dispatched issues to avoid duplicates
EXISTING=$(gh issue list -R "$REPO" --label jules --state all --limit 200 --json title -q '.[].title' 2>/dev/null)

# Get simple beads issues
CANDIDATES=$(bd list --status=open 2>/dev/null | grep -iE "$PATTERN" | sed 's/^[○◐●✓❄] //' | head -100)

dispatched=0
skipped=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ $dispatched -ge $MAX_DISPATCH ]] && break

    # Extract beads ID and title
    beads_id=$(echo "$line" | awk '{print $1}')
    # Remove priority marker and extract title
    title=$(echo "$line" | sed "s/^${beads_id} [●○◐] P[0-4] //")

    # Skip if already dispatched
    if echo "$EXISTING" | grep -qF "$title"; then
        skipped=$((skipped + 1))
        continue
    fi

    # Get full description from beads
    description=$(bd show "$beads_id" 2>/dev/null | sed -n '/^DESCRIPTION$/,/^$/p' | tail -n +2 | head -20)
    if [[ -z "$description" ]]; then
        description="Remove or fix: $title"
    fi

    body="**Beads ID:** \`$beads_id\`

$description

---
*Auto-dispatched from beads backlog by \`dispatch-jules.sh\`.*
*Quality gates: \`cargo fmt\`, \`cargo clippy --workspace -- -D warnings\`, \`cargo check --workspace\`, \`cargo test\`.*"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would create: $title (beads: $beads_id)"
    else
        url=$(gh issue create -R "$REPO" --title "$title" --body "$body" --label jules 2>&1)
        echo "Created: $url — $title (beads: $beads_id)"
    fi
    dispatched=$((dispatched + 1))

done <<< "$CANDIDATES"

echo ""
echo "Dispatched: $dispatched | Skipped (already exists): $skipped | Max: $MAX_DISPATCH"
