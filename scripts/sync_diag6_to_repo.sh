#!/usr/bin/env bash
# Copy Diag 6 outputs from WSL into the repo's results/ tree and the paper/.
set -u
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
DIAG_SRC=$HOME/ece567/diagnostics

for s in 0 1 2; do
  src="$DIAG_SRC/phase3b_residual_overlay_seed${s}"
  dst="$REPO/results/antmaze-teleport-navigate-v0/ex_chiql_phase3b/seed${s}/diagnostics"
  mkdir -p "$dst"
  if [ -d "$src" ]; then
    for f in sigma_residual_stats.json sigma_residual_overlay.png; do
      if [ -f "$src/$f" ]; then
        cp "$src/$f" "$dst/$f"
        echo "seed${s}: $f OK"
      fi
    done
  fi
done

# Copy seed 0's figure into paper/ for inclusion in §6.2.
cp "$DIAG_SRC/phase3b_residual_overlay_seed0/sigma_residual_overlay.png" \
   "$REPO/paper/diag6_residual_overlay.png"
echo "paper/ image OK"
