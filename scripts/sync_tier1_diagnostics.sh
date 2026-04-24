#!/usr/bin/env bash
# Pull all Tier-1 diagnostic outputs from WSL into the repo's results/ tree.
#
# Syncs:
#   - scoring_variants CSV per seed per config
#   - beta_sweep CSV per seed (shared-trunk EX-HIQL only so far)
#   - sigma_stats.json + sigma_diagnostic.png per seed (shared-trunk only)
#   - sigma_teleport_stats.json + sigma_teleport_overlay.png per seed for both
set -u
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
DIAG_SRC=$HOME/ece567/diagnostics

# --- per config/seed ---
for pair in \
    "shared_trunk_ex_chiql:shared_trunk_ex_chiql" \
    "ex_chiql_phase3b:ex_chiql_phase3b"; do
  RUN_TAG=${pair%%:*}
  RES_TAG=${pair##*:}
  RESULTS_BASE=$REPO/results/antmaze-teleport-navigate-v0/$RES_TAG
  RUN_BASE=$HOME/ece567/runs/$RUN_TAG

  echo "================ $RUN_TAG -> $RES_TAG ================"
  for s in 0 1 2; do
    dst=$RESULTS_BASE/seed${s}/diagnostics
    mkdir -p "$dst"

    # Scoring variants CSV
    src="$RUN_BASE/scoring_variants/scoring_variants_seed${s}.csv"
    if [ -f "$src" ]; then
      cp "$src" "$dst/scoring_variants.csv"
      echo "  seed$s scoring_variants.csv OK"
    fi

    # β sweep CSV (only shared-trunk_ex_chiql has it currently)
    src="$RUN_BASE/beta_sweep/beta_sweep_seed${s}.csv"
    if [ -f "$src" ]; then
      cp "$src" "$dst/beta_sweep.csv"
      echo "  seed$s beta_sweep.csv OK"
    fi

    # σ diagnostic outputs (for shared-trunk EX-HIQL only — we ran this one)
    for f in sigma_stats.json sigma_diagnostic.png; do
      src="$DIAG_SRC/shared_trunk_ex_seed${s}/$f"
      if [ "$RUN_TAG" = "shared_trunk_ex_chiql" ] && [ -f "$src" ]; then
        cp "$src" "$dst/$f"
        echo "  seed$s $f OK"
      fi
    done

    # Teleport-overlay outputs (for both configs)
    for f in sigma_teleport_stats.json sigma_teleport_overlay.png; do
      src="$DIAG_SRC/${RUN_TAG}_teleport_overlay_seed${s}/$f"
      if [ -f "$src" ]; then
        cp "$src" "$dst/$f"
        echo "  seed$s $f OK"
      fi
    done
  done
done
