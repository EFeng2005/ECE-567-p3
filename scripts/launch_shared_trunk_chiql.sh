#!/usr/bin/env bash
# Launch 3 parallel seeds of shared-trunk C-HIQL on antmaze-teleport-navigate-v0.
# Each seed reuses run_shared_trunk_chiql_local.sh; they share the GPU via
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 set in that script.
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
OGB="$REPO/external/ogbench_full/impls"

# Ensure the tracked shared-trunk chiql.py is in place before launching.
cp "$REPO/external/ogbench/impls/agents/chiql.py" "$OGB/agents/chiql.py"

for s in 0 1 2; do
  echo "launching seed=$s"
  setsid nohup bash "$REPO/scripts/run_shared_trunk_chiql_local.sh" "$s" \
    < /dev/null > /dev/null 2>&1 &
  echo "  pid=$!"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep -E 'run_shared_trunk_chiql_local|main.py' | grep -v grep || echo '(none)'
