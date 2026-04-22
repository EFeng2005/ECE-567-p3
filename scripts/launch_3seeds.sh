#!/usr/bin/env bash
# Launch 3 C-HIQL seeds in parallel, detached (survives shell exit).
set -euo pipefail

RUNNER="$HOME/ece567/run_chiql_local.sh"
LOGDIR="$HOME/ece567/logs"
mkdir -p "$LOGDIR"

for seed in 0 1 2; do
  setsid nohup bash "$RUNNER" "$seed" >/dev/null 2>&1 < /dev/null &
  pid=$!
  echo "launched seed=$seed pid=$pid"
  sleep 30
done

sleep 5
echo "--- ps ---"
ps -ef | grep main.py | grep -v grep || echo "(no main.py processes found)"
