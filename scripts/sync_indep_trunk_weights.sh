#!/usr/bin/env bash
# Sync all indep-trunk-5head checkpoints (Phase-3a + Phase-3b) + their CSVs
# from WSL into the repo's results/ tree. Idempotent.
set -u
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
ROOT_DST=$REPO/results/antmaze-teleport-navigate-v0

# (src_parent, dst_name) pairs.
for pair in \
    "ex_chiql_phase3:ex_chiql_phase3_design_a" \
    "ex_chiql_phase3b:ex_chiql_phase3b"; do
  src_parent=${pair%%:*}
  dst_name=${pair##*:}
  SRC_BASE=/home/y_f/ece567/runs/$src_parent/dummy/$src_parent
  DST_BASE=$ROOT_DST/$dst_name
  echo "================ $src_parent -> $dst_name ================"
  for s in 0 1 2; do
    src=$(ls -d $SRC_BASE/sd00${s}_* 2>/dev/null | head -1)
    dst=$DST_BASE/seed${s}
    mkdir -p "$dst"
    if [ -z "$src" ]; then echo "  seed${s}: (no src dir)"; continue; fi
    echo "  seed${s}: $src"
    for f in params_1000000.pkl train.csv eval.csv flags.json; do
      if [ -f "$src/$f" ]; then
        cp "$src/$f" "$dst/$f"
        sz=$(stat -c %s "$dst/$f")
        mb=$(awk -v s=$sz 'BEGIN{printf "%.1f", s/1048576}')
        echo "    $f -> ${mb} MiB"
      fi
    done
    echo "$src" > "$dst/source.txt"
  done
done
