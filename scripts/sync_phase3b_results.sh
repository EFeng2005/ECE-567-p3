#!/usr/bin/env bash
# Copy Phase-3b eval/train CSVs and flags.json from WSL Linux FS into the
# repo's results/ tree, matching the Phase-1 layout.
# Safe to re-run — overwrites the copied files, idempotent.
set -u

BASE=/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b
DST=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567/results/antmaze-teleport-navigate-v0/ex_chiql_phase3b

for s in 0 1 2; do
  src=$(ls -d "$BASE"/sd00${s}_* 2>/dev/null | head -1)
  dst="$DST/seed$s"
  mkdir -p "$dst"
  if [ -z "$src" ]; then
    echo "seed$s: source dir not found, skipping"
    continue
  fi
  echo "seed$s: $src  →  $dst"
  for f in eval.csv train.csv flags.json; do
    if [ -f "$src/$f" ]; then
      cp "$src/$f" "$dst/$f"
      echo "  $f ✓"
    else
      echo "  $f (missing in source)"
    fi
  done
  echo "$src" > "$dst/source.txt"
done
echo
echo "Note: params_1000000.pkl NOT copied (too large for git; kept on WSL for β-sweep / diagnostic)."
