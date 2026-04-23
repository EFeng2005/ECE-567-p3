#!/usr/bin/env bash
# Report per-seed step count and throughput across all active runs.
set -u
ROOT=/home/y_f/ece567/runs

for tag in shared_trunk_ex_chiql ex_chiql_wide_clip; do
  BASE=$ROOT/$tag/dummy/$tag
  echo "=== $tag ==="
  for s in 0 1 2; do
    d=$(ls -d $BASE/sd00${s}_* 2>/dev/null | head -1)
    if [ -n "$d" ]; then
      step=$(tail -1 $d/train.csv 2>/dev/null | awk -F, '{print $NF}')
      # Age of the directory for rate calc
      mtime=$(stat -c %Y $d/train.csv 2>/dev/null || echo 0)
      ctime=$(stat -c %Y $d 2>/dev/null || echo 0)
      elapsed=$((mtime - ctime))
      if [ $elapsed -gt 0 ] && [ "$step" -gt 0 ] 2>/dev/null; then
        rate=$(awk -v s=$step -v t=$elapsed 'BEGIN{printf "%.1f", s/t}')
        remain=$(awk -v s=$step -v t=$elapsed 'BEGIN{if (s>0) printf "%.0f", (1000000-s)*t/s; else print "?"}')
        echo "  seed$s step=$step elapsed=${elapsed}s rate=${rate} it/s eta=${remain}s"
      else
        echo "  seed$s step=$step (no timing yet)"
      fi
    else
      echo "  seed$s (dir missing)"
    fi
  done
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.free --format=csv,noheader
