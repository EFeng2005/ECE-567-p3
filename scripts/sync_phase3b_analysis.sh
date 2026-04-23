#!/usr/bin/env bash
# Copy the Phase-3b σ-diagnostic outputs AND β-sweep CSVs (if present)
# from WSL Linux FS into the repo's results/ tree. Idempotent.
set -u

DIAG_SRC=/home/y_f/ece567/diagnostics
SWEEP_SRC=/home/y_f/ece567/runs/ex_chiql_phase3b/beta_sweep
DST_ROOT=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567/results/antmaze-teleport-navigate-v0/ex_chiql_phase3b

# --- diagnostic outputs (sigma_stats.json + sigma_diagnostic.png per seed) ---
for s in 0 1 2; do
  src="$DIAG_SRC/ex3b_seed$s"
  dst="$DST_ROOT/seed$s/diagnostics"
  mkdir -p "$dst"
  if [ -f "$src/sigma_stats.json" ]; then
    cp "$src/sigma_stats.json" "$dst/sigma_stats.json"
    echo "seed$s: diagnostics/sigma_stats.json ✓"
  fi
  if [ -f "$src/sigma_diagnostic.png" ]; then
    cp "$src/sigma_diagnostic.png" "$dst/sigma_diagnostic.png"
    echo "seed$s: diagnostics/sigma_diagnostic.png ✓"
  fi
done
echo

# --- β-sweep CSVs (one per seed, written at end of each seed's full sweep) ---
mkdir -p "$DST_ROOT/beta_sweep"
for s in 0 1 2; do
  src="$SWEEP_SRC/beta_sweep_seed${s}.csv"
  dst="$DST_ROOT/beta_sweep/beta_sweep_seed${s}.csv"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "β-sweep seed$s: $dst ✓"
  else
    echo "β-sweep seed$s: still running (no CSV yet)"
  fi
done
echo
echo "Re-run after the β-sweep finishes to pick up the CSVs."
