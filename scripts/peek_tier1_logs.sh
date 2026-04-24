#!/usr/bin/env bash
# Peek at the tail of each Tier-1 diagnostic's log.
set -u
for s in 0 1 2; do
  echo "=== seed $s β-sweep ==="
  tail -5 "$HOME/ece567/logs/beta_sweep_shared_trunk_ex_seed${s}.log" 2>/dev/null || echo "(no log)"
  echo
  echo "=== seed $s σ-diag ==="
  tail -5 "$HOME/ece567/logs/diag_shared_trunk_ex_seed${s}.log" 2>/dev/null || echo "(no log)"
  echo
done

echo "--- process list ---"
ps -ef | grep -E 'eval_beta_sweep_exhiql|diagnose_sigma_exhiql' | grep -v grep | awk '{print $2, $5, $10}'
