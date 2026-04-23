#!/usr/bin/env bash
# Single-seed shared-trunk EX-HIQL run (WSL) on antmaze-teleport-navigate-v0.
# Same Phase-3b config (tight head_expectiles=(0.6, 0.65, 0.7, 0.75, 0.8),
# Design A actors pinned to head 2 = τ=0.7), but on the shared-trunk-5head
# branch where ex_chiql.py's value network is SharedTrunkGCValue (1 shared
# trunk + 5 Dense(1) heads).
#
# Usage: bash scripts/run_shared_trunk_ex_chiql_local.sh <seed>
set -euo pipefail

SEED="${1:?usage: run_shared_trunk_ex_chiql_local.sh <seed>}"
REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

# Sync the tracked shared-trunk ex_chiql.py into the untracked ogbench_full
# working copy. Required per-branch because ogbench_full doesn't flip with
# git checkout (it's an untracked upstream clone used at runtime).
cp "$REPO/external/ogbench/impls/agents/ex_chiql.py" "$OGB/agents/ex_chiql.py"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/shared_trunk_ex_chiql"
LOG_DIR="$HOME/ece567/logs"
mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR" "$LOG_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

cd "$OGB"

LOG_FILE="$LOG_DIR/shared_trunk_ex_chiql_seed${SEED}.log"
echo "Starting shared-trunk EX-HIQL seed=$SEED. Log: $LOG_FILE"

"$VENV/bin/python" -u main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/ex_chiql.py \
  --seed="$SEED" \
  --run_group=shared_trunk_ex_chiql \
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
