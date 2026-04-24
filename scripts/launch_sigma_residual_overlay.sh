#!/usr/bin/env bash
# Run Diag 6 (residual σ⊥ teleport overlay) on all 3 Phase-3b seeds.
# Serial execution; each run is ~1-2 min CPU.
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
VENV=$HOME/ece567/.venv
OGB=$REPO/external/ogbench_full/impls
LOG_DIR=$HOME/ece567/logs
mkdir -p "$LOG_DIR"

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

# Phase-3b checkpoints live on indep-trunk-5head; sync its ex_chiql.py into
# ogbench_full so python loads the indep-trunks variant.
git -C "$REPO" show indep-trunk-5head:external/ogbench/impls/agents/ex_chiql.py \
    > "$OGB/agents/ex_chiql.py"

BASE=/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b
cd "$OGB"
for s in 0 1 2; do
  d=$(ls -d "$BASE/sd00${s}_"* 2>/dev/null | head -1)
  ckpt="$d/params_1000000.pkl"
  [ -f "$ckpt" ] || { echo "seed $s: no checkpoint"; continue; }
  out=$HOME/ece567/diagnostics/phase3b_residual_overlay_seed${s}
  log=$LOG_DIR/diag_residual_seed${s}.log
  echo "=== seed $s ==="
  "$VENV/bin/python" -u "$REPO/scripts/diagnose_sigma_residual_overlay.py" \
    --seed "$s" --ckpt "$ckpt" --out_dir "$out" --n_states 2000 \
    > "$log" 2>&1
  if [ $? -eq 0 ]; then
    echo "  OK -> $out"
    grep -E 'residual fit|ratio_median|dist_sigma_corr' "$out/sigma_residual_stats.json" 2>/dev/null || true
  else
    echo "  FAILED -> see $log"
    tail -20 "$log"
  fi
done
