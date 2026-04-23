#!/usr/bin/env bash
# Tier-1 diagnostics for the shared-trunk EX-HIQL run (3 seeds, step-1M):
#   - β sweep at β ∈ {0, 0.25, 0.5, 1, 2, 4} per seed
#   - σ diagnostic per seed (σ stats + overlap at each β)
# Both are eval-only (CPU-bound, no backprop), so they can run in parallel.
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
OGB=$REPO/external/ogbench_full/impls
VENV=$HOME/ece567/.venv

# Sync the shared-trunk ex_chiql.py (with SharedTrunkGCValue) into ogbench_full
# so python's `agents.ex_chiql` picks up the correct architecture.
cp "$REPO/external/ogbench/impls/agents/ex_chiql.py" "$OGB/agents/ex_chiql.py"

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql
OUT_SWEEP=/home/y_f/ece567/runs/shared_trunk_ex_chiql/beta_sweep
OUT_DIAG_ROOT=/home/y_f/ece567/diagnostics
LOG_DIR=$HOME/ece567/logs
mkdir -p "$OUT_SWEEP" "$OUT_DIAG_ROOT" "$LOG_DIR"

cd "$OGB"

for s in 0 1 2; do
  d=$(ls -d "$BASE/sd00${s}_"* 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "skipping seed=$s: no checkpoint dir"; continue; fi
  ckpt="$d/params_1000000.pkl"
  if [ ! -f "$ckpt" ]; then echo "skipping seed=$s: no params_1000000.pkl"; continue; fi

  # β sweep
  sweep_out="$OUT_SWEEP/beta_sweep_seed${s}.csv"
  sweep_log="$LOG_DIR/beta_sweep_shared_trunk_ex_seed${s}.log"
  setsid nohup "$VENV/bin/python" -u \
    "$REPO/scripts/eval_beta_sweep_exhiql.py" \
    --seed "$s" --ckpt "$ckpt" --out "$sweep_out" \
    > "$sweep_log" 2>&1 < /dev/null &
  echo "launched β-sweep seed=$s pid=$! log=$sweep_log"

  # σ diagnostic
  diag_out="$OUT_DIAG_ROOT/shared_trunk_ex_seed${s}"
  diag_log="$LOG_DIR/diag_shared_trunk_ex_seed${s}.log"
  setsid nohup "$VENV/bin/python" -u \
    "$REPO/scripts/diagnose_sigma_exhiql.py" \
    --seed "$s" --ckpt "$ckpt" --out_dir "$diag_out" \
    > "$diag_log" 2>&1 < /dev/null &
  echo "launched σ-diag  seed=$s pid=$! log=$diag_log"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep -E 'eval_beta_sweep_exhiql|diagnose_sigma_exhiql' | grep -v grep | awk '{print $2, $9, $10, $11}' | head -20
