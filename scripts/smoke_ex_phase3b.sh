#!/usr/bin/env bash
# 500-step smoke test for EX-HIQL Phase-3b (tight head_expectiles).
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV_LINUX="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/smoke_ex_phase3b"
mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$OGB"

"$VENV_LINUX/bin/python" main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/ex_chiql.py \
  --seed=0 \
  --run_group=smoke_ex_phase3b \
  --save_dir="$SAVE_DIR" \
  --dataset_dir="$OGBENCH_DATASET_DIR" \
  --wandb_mode=disabled \
  --train_steps=500 \
  --log_interval=100 \
  --eval_interval=500 \
  --save_interval=500 \
  --eval_episodes=3 \
  --video_episodes=0 \
  --agent.num_value_heads=5 \
  --agent.num_subgoal_candidates=16 \
  --agent.pessimism_beta=0.5 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0
