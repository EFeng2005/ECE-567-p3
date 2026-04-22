#!/usr/bin/env bash
# Run the C-HIQL beta sweep on all 3 saved checkpoints in parallel (one process
# per seed). Each process loops through BETAS = [0, 0.25, 0.5, 1.0, 2.0] and
# writes ~/ece567/runs/chiql_phase2/beta_sweep_seed{N}.csv.
set -euo pipefail

VENV="$HOME/ece567/.venv"
REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
OGB="$REPO/external/ogbench_full/impls"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# All eval on CPU, so JAX doesn't need much GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

RUNS_BASE="/home/y_f/ece567/runs/chiql_phase2/dummy/chiql_phase2"
OUT_DIR="/home/y_f/ece567/runs/chiql_phase2/beta_sweep"
LOG_DIR="/home/y_f/ece567/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

declare -A SEED_DIRS=(
  [0]="$RUNS_BASE/sd000_20260421_235933"
  [1]="$RUNS_BASE/sd001_20260422_000000"
  [2]="$RUNS_BASE/sd002_20260422_000030"
)

cd "$OGB"
for seed in 0 1 2; do
  ckpt="${SEED_DIRS[$seed]}/params_1000000.pkl"
  out="$OUT_DIR/beta_sweep_seed${seed}.csv"
  log="$LOG_DIR/beta_sweep_seed${seed}.log"
  setsid nohup "$VENV/bin/python" -u "$REPO/scripts/eval_beta_sweep.py" \
    --seed "$seed" --ckpt "$ckpt" --out "$out" \
    > "$log" 2>&1 < /dev/null &
  pid=$!
  echo "launched seed=$seed pid=$pid log=$log"
  sleep 10
done

sleep 5
echo "--- ps ---"
ps -ef | grep eval_beta_sweep | grep -v grep || echo "(none)"
