#!/usr/bin/env bash
# Launch 3 EX-HIQL seeds in parallel, detached (survives shell exit).
# Design A defaults. For Design B set EX_EXTRA below.
set -euo pipefail

RUNNER="$HOME/ece567/run_ex_chiql_local.sh"
LOGDIR="$HOME/ece567/logs"
mkdir -p "$LOGDIR"

EX_EXTRA=""   # e.g. "--agent.actor_expectile_index_high=2" for Design B

for seed in 0 1 2; do
  setsid nohup bash "$RUNNER" "$seed" $EX_EXTRA >/dev/null 2>&1 < /dev/null &
  pid=$!
  echo "launched seed=$seed pid=$pid"
  sleep 30
done

sleep 5
echo "--- ps ---"
ps -ef | grep main.py | grep -v grep || echo "(no main.py processes found)"
