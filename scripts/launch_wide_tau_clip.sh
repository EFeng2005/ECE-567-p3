#!/usr/bin/env bash
# Launch 6 parallel seeds of EX-HIQL-clip on two τ configurations:
#   - wide = (0.1, 0.3, 0.5, 0.7, 0.9)  (Phase-3a config, exploded without clipping)
#   - mid  = (0.4, 0.5, 0.6, 0.7, 0.8)  (moderate extension of Phase-3b)
# 3 seeds per config = 6 total. With the 3 concurrent shared-trunk EX-HIQL
# seeds already running, this brings the GPU load to 9 concurrent processes.
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
OGB="$REPO/external/ogbench_full/impls"

# Sync tracked ex_chiql_clip.py before launch.
cp "$REPO/external/ogbench/impls/agents/ex_chiql_clip.py" "$OGB/agents/ex_chiql_clip.py"

for tag in wide mid; do
  for s in 0 1 2; do
    echo "launching tag=$tag seed=$s"
    setsid nohup bash "$REPO/scripts/run_wide_tau_clip_local.sh" "$s" "$tag" \
      < /dev/null > /dev/null 2>&1 &
    echo "  pid=$!"
  done
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep -E 'run_wide_tau_clip_local|main.py' | grep -v grep || echo '(none)'
echo
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.free --format=csv,noheader
