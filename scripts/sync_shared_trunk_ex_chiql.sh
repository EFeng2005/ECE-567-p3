#!/usr/bin/env bash
# Sync shared-trunk EX-HIQL training outputs (train.csv, eval.csv,
# flags.json, params_1000000.pkl) from WSL into the repo's results tree.
# Idempotent — safe to re-run after each seed finishes.
set -u
BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
DST=$REPO/results/antmaze-teleport-navigate-v0/shared_trunk_ex_chiql

for s in 0 1 2; do
  src=$(ls -d $BASE/sd00${s}_* 2>/dev/null | head -1)
  dst=$DST/seed${s}
  mkdir -p "$dst"
  if [ -z "$src" ]; then echo "seed${s}: (no src)"; continue; fi
  echo "seed${s}: $src"
  for f in params_1000000.pkl train.csv eval.csv flags.json; do
    if [ -f "$src/$f" ]; then
      cp "$src/$f" "$dst/$f"
      sz=$(stat -c %s "$dst/$f")
      mb=$(awk -v s=$sz 'BEGIN{printf "%.1f", s/1048576}')
      echo "  $f -> ${mb} MiB"
    fi
  done
  echo "$src" > "$dst/source.txt"
done
