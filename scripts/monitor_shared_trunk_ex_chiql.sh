#!/usr/bin/env bash
# Status report for the shared-trunk EX-HIQL 3-seed run.
set -u
BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
echo
echo "=== procs ==="
ps -o pid,etime,pcpu,rss,comm -p "$(pgrep -d, -f main.py 2>/dev/null)" 2>/dev/null | head -10
echo

for s in 0 1 2; do
  d=$(ls -d "$BASE"/sd00${s}_* 2>/dev/null | head -1)
  echo "=== seed$s ($d) ==="
  if [ -z "$d" ]; then echo "  (dir missing)"; continue; fi
  echo "--- train.csv last row ---"
  tail -1 "$d/train.csv" 2>/dev/null || echo "(no train.csv yet)"
  echo "--- eval.csv (all rows) ---"
  cat "$d/eval.csv" 2>/dev/null || echo "(no eval.csv yet)"
  echo
done

all_done=1
for s in 0 1 2; do
  d=$(ls -d "$BASE"/sd00${s}_* 2>/dev/null | head -1)
  if [ -z "$d" ]; then all_done=0; break; fi
  grep -q ",1000000$" "$d/train.csv" 2>/dev/null || { all_done=0; break; }
  grep -q ",1000000$" "$d/eval.csv" 2>/dev/null || { all_done=0; break; }
  [ -f "$d/params_1000000.pkl" ] || { all_done=0; break; }
done
if [ "$all_done" = "1" ]; then
  echo "=== ALL THREE SEEDS COMPLETE ==="
fi
