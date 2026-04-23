#!/usr/bin/env bash
# Launch 3 parallel seeds of shared-trunk EX-HIQL.
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
OGB="$REPO/external/ogbench_full/impls"

# Sync tracked ex_chiql.py (shared-trunk variant on this branch) before launch.
cp "$REPO/external/ogbench/impls/agents/ex_chiql.py" "$OGB/agents/ex_chiql.py"

for s in 0 1 2; do
  echo "launching seed=$s"
  setsid nohup bash "$REPO/scripts/run_shared_trunk_ex_chiql_local.sh" "$s" \
    < /dev/null > /dev/null 2>&1 &
  echo "  pid=$!"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep -E 'run_shared_trunk_ex_chiql_local|main.py' | grep -v grep || echo '(none)'
