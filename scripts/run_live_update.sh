#!/usr/bin/env bash
# Process supervision for the forecast engine's live_update loop.
#
# live_update.py has no supervision of its own — if it dies (uncaught
# exception, killed process, machine hiccup), nothing restarts it and
# forecasts silently go stale with no alert. This wrapper restarts it
# on any exit, with a short cooldown so a fast-failing process (e.g. a
# bad .env) doesn't spin the CPU in a tight crash loop.
#
# Usage:
#   ./scripts/run_live_update.sh                  # run in foreground
#   nohup ./scripts/run_live_update.sh &          # run detached
#   (or install scripts/launchd/*.plist for auto-start-on-login + auto-restart)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
SUPERVISOR_LOG="$LOG_DIR/live_update_supervisor.log"
RESTART_COOLDOWN_SECONDS=5

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1" >> "$SUPERVISOR_LOG"
}

cd "$PROJECT_ROOT"
log "supervisor started (pid $$)"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

while true; do
    log "starting live_update.py"
    "$PROJECT_ROOT/venv/bin/python" -m swdss.features.live_update
    exit_code=$?
    log "live_update.py exited with code $exit_code — restarting in ${RESTART_COOLDOWN_SECONDS}s"
    sleep "$RESTART_COOLDOWN_SECONDS"
done
