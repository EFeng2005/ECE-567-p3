#!/usr/bin/env bash
# Quick sanity test: run teleport overlay on ONE seed of ONE config, to
# make sure the script works before committing to all 6.
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
VENV=$HOME/ece567/.venv
OGB=$REPO/external/ogbench_full/impls
LOG_DIR=$HOME/ece567/logs

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

# Pick shared_trunk_ex_chiql seed 0 for the sanity test.
git -C "$REPO" show "shared-trunk-5head:external/ogbench/impls/agents/ex_chiql.py" > "$OGB/agents/ex_chiql.py"
BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql
d=$(ls -d $BASE/sd000_* 2>/dev/null | head -1)
ckpt="$d/params_1000000.pkl"
out=$HOME/ece567/diagnostics/shared_trunk_ex_chiql_teleport_overlay_seed0_TEST
log=$LOG_DIR/diag_teleport_test.log

echo "running: $ckpt -> $out"
cd "$OGB"
"$VENV/bin/python" -u "$REPO/scripts/diagnose_sigma_teleport_overlay.py" \
  --seed 0 --ckpt "$ckpt" --out_dir "$out" --n_states 500 \
  2>&1 | tee "$log"

echo
echo "=== output files ==="
ls -la "$out" 2>/dev/null
