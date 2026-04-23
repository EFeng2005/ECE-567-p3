#!/usr/bin/env bash
# Run a single shared-trunk C-HIQL seed (WSL) on antmaze-teleport-navigate-v0.
# Same config as Phase-2 C-HIQL (scripts/run_chiql_local.sh), same agent name
# (agents/chiql.py), but on the `shared-trunk-chiql` branch where chiql.py
# has been patched to use SharedTrunkGCValue (1 shared trunk + 5 Dense(1)
# heads) instead of GCValue(ensemble=True) (5 independent MLPs).
#
# Usage: bash scripts/run_shared_trunk_chiql_local.sh <seed>
set -euo pipefail

SEED="${1:?usage: run_shared_trunk_chiql_local.sh <seed>}"
REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

# Sync our tracked chiql.py into the untracked ogbench_full working copy
# so the python process actually runs the shared-trunk version. This is
# required per-branch because ogbench_full is untracked and doesn't flip
# with git checkout.
cp "$REPO/external/ogbench/impls/agents/chiql.py" "$OGB/agents/chiql.py"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/shared_trunk_chiql"
LOG_DIR="$HOME/ece567/logs"
mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR" "$LOG_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

cd "$OGB"

LOG_FILE="$LOG_DIR/shared_trunk_chiql_seed${SEED}.log"
echo "Starting shared-trunk C-HIQL seed=$SEED. Log: $LOG_FILE"

"$VENV/bin/python" -u main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/chiql.py \
  --seed="$SEED" \
  --run_group=shared_trunk_chiql \
  --save_dir="$SAVE_DIR" \
  --dataset_dir="$OGBENCH_DATASET_DIR" \
  --wandb_mode=disabled \
  --train_steps=1000000 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=1000000 \
  --eval_episodes=50 \
  --video_episodes=0 \
  --agent.num_value_heads=5 \
  --agent.num_subgoal_candidates=16 \
  --agent.pessimism_beta=0.5 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  >>"$LOG_FILE" 2>&1
