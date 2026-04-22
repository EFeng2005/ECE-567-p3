#!/usr/bin/env bash
# Tiny C-HIQL smoke test inside WSL — verifies dataset download, agent build,
# JIT compilation, and one eval rollout work end-to-end.
set -euo pipefail

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV="$REPO/.venv_wsl"   # not used — we use the Linux-FS venv below
VENV_LINUX="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

# Dataset and save dir on Linux FS for speed
export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/smoke"

mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR"

# MuJoCo: WSL has EGL available via nvidia driver passthrough
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$OGB"

"$VENV_LINUX/bin/python" main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/chiql.py \
  --seed=0 \
  --run_group=smoke \
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
