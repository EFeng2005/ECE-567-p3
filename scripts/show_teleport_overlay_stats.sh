#!/usr/bin/env bash
# Dump sigma_teleport_stats.json from every teleport overlay run.
set -u
for cfg in shared_trunk_ex_chiql ex_chiql_phase3b; do
  for s in 0 1 2; do
    out="$HOME/ece567/diagnostics/${cfg}_teleport_overlay_seed${s}/sigma_teleport_stats.json"
    echo "=== ${cfg} seed${s} ==="
    if [ -f "$out" ]; then
      python3 -c "
import json
d = json.load(open('$out'))
near = d['by_category'].get('near', {})
fr = d['by_category'].get('far', {})
print(f'  n_near={near.get(\"n_states\",0):3d}  n_far={fr.get(\"n_states\",0):4d}')
print(f'  near_σ_mean={near.get(\"sigma_mean\",0):.2f}  far_σ_mean={fr.get(\"sigma_mean\",0):.2f}')
print(f'  near_σ_median={near.get(\"sigma_median\",0):.2f}  far_σ_median={fr.get(\"sigma_median\",0):.2f}')
print(f'  ratio_mean={d[\"near_vs_far_ratio_mean\"]:.3f}  ratio_median={d[\"near_vs_far_ratio_median\"]:.3f}')
print(f'  dist_to_teleport_vs_σ_corr={d[\"dist_to_teleport_vs_sigma_corr\"]:.3f}')
"
    else
      echo "  (missing)"
    fi
  done
done
echo
echo "=== β-sweep CSVs (shared_trunk_ex_chiql) ==="
for s in 0 1 2; do
  csv="$HOME/ece567/runs/shared_trunk_ex_chiql/beta_sweep/beta_sweep_seed${s}.csv"
  if [ -f "$csv" ]; then
    echo "--- seed $s ---"
    cat "$csv"
    echo
  fi
done
