#!/usr/bin/env bash
# Run a single C-HIQL seed locally (inside WSL) on antmaze-teleport-navigate-v0.
# Mirrors scripts/submit_chiql.sh but invokes python directly instead of sbatch.
#
# Usage: bash scripts/run_chiql_local.sh <seed>
set -euo pipefail

SEED="${1:?usage: run_chiql_local.sh <seed>}"
REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/chiql_phase2"
LOG_DIR="$HOME/ece567/logs"
mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR" "$LOG_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Share the GPU across seeds — tell XLA not to pre-allocate 75% of VRAM.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

cd "$OGB"

LOG_FILE="$LOG_DIR/chiql_seed${SEED}.log"
echo "Starting C-HIQL seed=$SEED. Log: $LOG_FILE"

"$VENV/bin/python" -u main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/chiql.py \
  --seed="$SEED" \
  --run_group=chiql_phase2 \
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
