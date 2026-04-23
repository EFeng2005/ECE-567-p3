#!/usr/bin/env bash
# Copy shared-trunk CHIQL training outputs from WSL into the repo's
# results tree. Mirrors sync_phase3b_results.sh pattern.
set -u
BASE=/home/y_f/ece567/runs/shared_trunk_chiql/dummy/shared_trunk_chiql
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
DST=$REPO/results/antmaze-teleport-navigate-v0/shared_trunk_chiql

for s in 0 1 2; do
  src=$(ls -d $BASE/sd00${s}_* 2>/dev/null | head -1)
  dst=$DST/seed${s}
  mkdir -p "$dst"
  echo "seed${s}: ${src}"
  [ -f "$src/train.csv" ] && cp "$src/train.csv" "$dst/train.csv" && echo "  train.csv OK"
  [ -f "$src/eval.csv" ] && cp "$src/eval.csv" "$dst/eval.csv" && echo "  eval.csv OK"
  [ -f "$src/flags.json" ] && cp "$src/flags.json" "$dst/flags.json" && echo "  flags.json OK"
  echo "$src" > "$dst/source.txt"
done
