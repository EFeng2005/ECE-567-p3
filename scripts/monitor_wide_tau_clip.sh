#!/usr/bin/env bash
# Status report for both wide and mid τ clipped runs.
set -u
ROOT=/home/y_f/ece567/runs

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.free --format=csv,noheader
echo
echo "=== procs ==="
ps -o pid,etime,pcpu,rss,comm -p "$(pgrep -d, -f main.py 2>/dev/null)" 2>/dev/null | head -20
echo

for tag in wide mid; do
  BASE="$ROOT/ex_chiql_${tag}_clip/dummy/ex_chiql_${tag}_clip"
  echo "================ $tag (head_expectiles=$tag) ================"
  for s in 0 1 2; do
    d=$(ls -d "$BASE"/sd00${s}_* 2>/dev/null | head -1)
    echo "--- ${tag}-seed$s ($d) ---"
    if [ -z "$d" ]; then echo "  (dir missing)"; continue; fi
    echo "  train.csv last row:"
    tail -1 "$d/train.csv" 2>/dev/null || echo "  (no train.csv yet)"
    echo "  eval.csv:"
    cat "$d/eval.csv" 2>/dev/null || echo "  (no eval.csv yet)"
    echo
  done
done
