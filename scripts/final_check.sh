#!/usr/bin/env bash
BASE=/home/y_f/ece567/runs/chiql_phase2/dummy/chiql_phase2
for s in 0 1 2; do
  d=$(ls -d "$BASE"/sd00${s}_* | head -1)
  echo "=== seed$s ==="
  tail -1 "$d/eval.csv"
done
echo ---procs---
ps -ef | grep main.py | grep -v grep || echo "(no main.py procs)"
