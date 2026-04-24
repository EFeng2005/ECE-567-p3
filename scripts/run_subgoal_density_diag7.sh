#!/usr/bin/env bash
# Diag 7: scoring-filtered subgoal density on Phase-3b seed 0, task 1.
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

git -C "$REPO" show indep-trunk-5head:external/ogbench/impls/agents/ex_chiql.py \
    > "$OGB/agents/ex_chiql.py"

BASE=/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b
d=$(ls -d "$BASE/sd000_"* 2>/dev/null | head -1)
ckpt="$d/params_1000000.pkl"
[ -f "$ckpt" ] || { echo "no checkpoint at $ckpt"; exit 1; }

out=$HOME/ece567/diagnostics/phase3b_subgoal_density_diag7_seed0
log=$LOG_DIR/diag7_subgoal_density.log
echo "ckpt=$ckpt"
echo "out=$out"
cd "$OGB"
"$VENV/bin/python" -u "$REPO/scripts/diagnose_subgoal_density_filtered.py" \
  --seed 0 --ckpt "$ckpt" --out_dir "$out" \
  --task_id 1 --n_samples 1000 --top_k 50 --nn_bank_size 10000 \
  --beta_pes 0.5 \
  2>&1 | tee "$log"
