#!/bin/bash
# jules-dispatch-cron.sh — Daily centralized Jules dispatch from ai-proxy.
#
# Runs from ai-proxy where beads databases are accessible for all repos.
# Dispatches up to 40 Jules sessions/day across all configured repos.
#
# Crontab entry (on ai-proxy):
#   0 8 * * * /home/brian/code/beefcake-swarm/scripts/jules-dispatch-cron.sh
#
set -euo pipefail

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:$PATH"
source ~/.swarm-env 2>/dev/null

cd ~/code/beefcake-swarm
git pull --ff-only 2>/dev/null

mkdir -p ~/logs

exec python3 scripts/jules-dispatch.py --max 40 --delay 2.0 \
  >> ~/logs/jules-dispatch-$(date +%Y%m%d).log 2>&1
