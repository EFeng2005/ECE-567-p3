#!/usr/bin/env bash
# Dump sigma_stats.json for each seed of the shared-trunk EX-HIQL diagnostic.
set -u
for s in 0 1 2; do
  echo "=== seed${s} sigma_stats.json ==="
  cat "$HOME/ece567/diagnostics/shared_trunk_ex_seed${s}/sigma_stats.json" 2>/dev/null || echo "(missing)"
  echo
done
