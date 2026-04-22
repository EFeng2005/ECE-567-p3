#!/usr/bin/env bash
# Compact status report for the 3-seed C-HIQL run.
set -u
BASE=/home/y_f/ece567/runs/chiql_phase2/dummy/chiql_phase2

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
echo
echo "=== procs ==="
ps -o pid,etime,comm -p "$(pgrep -d, -f main.py 2>/dev/null)" 2>/dev/null | head -10
echo

for s in 0 1 2; do
  d=$(ls -d "$BASE"/sd00${s}_* 2>/dev/null | head -1)
  echo "=== seed$s ($d) ==="
  if [ -z "$d" ]; then echo "  (dir missing)"; continue; fi
  echo "--- train.csv last row ---"
  head -1 "$d/train.csv" | awk -F, '{for (i=1;i<=NF;i++) printf "[%d]%s ",i,$i; print ""}' | head -c 200
  echo
  tail -1 "$d/train.csv"
  echo "--- eval.csv (all rows) ---"
  cat "$d/eval.csv"
  echo
done
