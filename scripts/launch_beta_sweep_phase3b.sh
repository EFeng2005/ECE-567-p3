#!/usr/bin/env bash
# Run the β sweep on the 3 Phase-3b EX-HIQL checkpoints in parallel.
# Each process evaluates the same checkpoint at β ∈ {0, 0.25, 0.5, 1.0, 2.0}.
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
OGB=$REPO/external/ogbench_full/impls
VENV=$HOME/ece567/.venv

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

BASE=/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b
OUT_DIR=/home/y_f/ece567/runs/ex_chiql_phase3b/beta_sweep
LOG_DIR=$HOME/ece567/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$OGB"
for s in 0 1 2; do
  d=$(ls -d "$BASE/sd00${s}_"* | head -1)
  ckpt="$d/params_1000000.pkl"
  out="$OUT_DIR/beta_sweep_seed${s}.csv"
  log="$LOG_DIR/beta_sweep_phase3b_seed${s}.log"

  setsid nohup "$VENV/bin/python" -u "$REPO/scripts/eval_beta_sweep.py" \
    --seed "$s" --ckpt "$ckpt" --out "$out" \
    > "$log" 2>&1 < /dev/null &
  echo "launched seed=$s pid=$! log=$log"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep eval_beta_sweep | grep -v grep || echo '(none)'
