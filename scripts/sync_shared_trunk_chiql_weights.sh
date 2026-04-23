#!/usr/bin/env bash
# Copy shared-trunk CHIQL checkpoints (params_1000000.pkl) from WSL
# into results/. CSVs etc. were already synced by sync_shared_trunk_chiql.sh.
set -u
BASE=/home/y_f/ece567/runs/shared_trunk_chiql/dummy/shared_trunk_chiql
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
DST=$REPO/results/antmaze-teleport-navigate-v0/shared_trunk_chiql

for s in 0 1 2; do
  src=$(ls -d $BASE/sd00${s}_* 2>/dev/null | head -1)
  dst=$DST/seed${s}
  mkdir -p "$dst"
  echo "seed${s}: ${src}"
  if [ -f "$src/params_1000000.pkl" ]; then
    cp "$src/params_1000000.pkl" "$dst/params_1000000.pkl"
    sz=$(stat -c %s "$dst/params_1000000.pkl")
    mb=$(awk -v s=$sz 'BEGIN{printf "%.1f", s/1048576}')
    echo "  params_1000000.pkl -> ${mb} MiB"
  fi
done
