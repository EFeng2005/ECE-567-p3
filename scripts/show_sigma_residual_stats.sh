#!/usr/bin/env bash
# Dump the full Diag 6 sigma_residual_stats.json for all 3 seeds.
for s in 0 1 2; do
  echo "============ seed $s ============"
  cat "$HOME/ece567/diagnostics/phase3b_residual_overlay_seed${s}/sigma_residual_stats.json"
  echo
done
