#!/usr/bin/env bash
# teammate-idle.sh — TeammateIdle hook
# Checks for uncommitted changes and reminds teammate to commit.
set -euo pipefail

UNCOMMITTED=$(git status --porcelain 2>/dev/null || echo "")

if [ -n "$UNCOMMITTED" ]; then
  echo "WARNING: You have uncommitted changes. Please commit before going idle:"
  echo "$UNCOMMITTED"
  echo ""
  echo "Run: git add <files> && git commit -m 'wip: <description>'"
  exit 2
fi

echo "Working tree clean. OK to idle."
exit 0
