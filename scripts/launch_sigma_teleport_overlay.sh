#!/usr/bin/env bash
# Queue σ-with-teleport-overlay diagnostic for every available step-1M
# checkpoint (shared_trunk_ex_chiql, ex_chiql_phase3b), 3 seeds each.
# Writes sigma_teleport_stats.json + sigma_teleport_overlay.png per seed.
#
# Usage: bash scripts/launch_sigma_teleport_overlay.sh
#
# Each run is ~2-3 min on CPU (500-2000 states × 16 candidates forward pass,
# no env rollouts). We run them serially here so they don't contend with
# the Level-1 eval processes already in flight.
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
VENV=$HOME/ece567/.venv
OGB=$REPO/external/ogbench_full/impls
LOG_DIR=$HOME/ece567/logs

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

mkdir -p "$LOG_DIR"

for pair in \
    "shared_trunk_ex_chiql:shared-trunk-5head:shared_trunk_ex_chiql" \
    "ex_chiql_phase3b:indep-trunk-5head:ex_chiql_phase3b"; do
  TAG=${pair%%:*}
  rest=${pair#*:}
  BRANCH=${rest%%:*}
  RUNDIR=${rest#*:}

  echo "=================== $TAG (branch $BRANCH) ==================="
  # Sync the right ex_chiql.py
  git -C "$REPO" show "$BRANCH:external/ogbench/impls/agents/ex_chiql.py" > "$OGB/agents/ex_chiql.py"

  BASE=/home/y_f/ece567/runs/$RUNDIR/dummy/$RUNDIR
  for s in 0 1 2; do
    d=$(ls -d "$BASE/sd00${s}_"* 2>/dev/null | head -1)
    ckpt="$d/params_1000000.pkl"
    if [ ! -f "$ckpt" ]; then
      echo "  skipping seed=$s: no checkpoint"
      continue
    fi
    out="$HOME/ece567/diagnostics/${TAG}_teleport_overlay_seed${s}"
    log="$LOG_DIR/diag_teleport_${TAG}_seed${s}.log"
    echo "  seed $s ..."
    cd "$OGB"
    "$VENV/bin/python" -u "$REPO/scripts/diagnose_sigma_teleport_overlay.py" \
      --seed "$s" --ckpt "$ckpt" --out_dir "$out" --n_states 2000 \
      > "$log" 2>&1
    if [ $? -eq 0 ]; then
      echo "    done -> $out"
    else
      echo "    FAILED -> see $log"
    fi
  done
done
