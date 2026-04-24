#!/usr/bin/env bash
for s in 0 1 2; do
  echo "=== Phase-3b beta_sweep seed $s ==="
  cat "$HOME/ece567/runs/ex_chiql_phase3b/beta_sweep/beta_sweep_seed${s}.csv" 2>/dev/null || echo "(missing)"
  echo
done
