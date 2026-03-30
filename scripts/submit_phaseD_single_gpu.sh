#!/usr/bin/env bash
# Phase D, single-GPU variant:
#   scene-play-v0, puzzle-3x3-play-v0, powderworld-hard-play-v0
#   3 datasets x 5 methods x 3 seeds = 45 runs = 45 SLURM jobs
#
# This variant submits one experiment per job to improve schedulability on
# partitions where 2-GPU jobs sit in queue for much longer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_1GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench.sbatch"

ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
RUN_GROUP="${RUN_GROUP:-phaseD_single}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
elif [ -d "/scratch/chaijy_root/chaijy2" ]; then
  DEFAULT_DATASET_DIR="/scratch/chaijy_root/chaijy2/$USER/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

VENV_PATH="${VENV_PATH:-}"

if [ ! -f "$SBATCH_1GPU" ]; then
  echo "Missing SLURM template: $SBATCH_1GPU"
  exit 1
fi

submit_1gpu() {
  local job_name="$1"
  local seed="$2"
  local env_name="$3"
  local agent_path="$4"
  local extra_args="$5"
  local mem="${6:-32G}"
  local time="${7:-08:00:00}"

  local exports="SEED=$seed,RUN_GROUP=$RUN_GROUP,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,ENV_NAME=$env_name,AGENT_PATH=$agent_path,EXTRA_ARGS=$extra_args"

  echo "  $job_name: $env_name + $(basename "$agent_path" .py) [seed=$seed]"
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name="$job_name" \
    --mem="$mem" --time="$time" --export="ALL,$exports" "$SBATCH_1GPU"
}

declare -A AGENTS=(
  [crl]="agents/crl.py"
  [hiql]="agents/hiql.py"
  [qrl]="agents/qrl.py"
  [gciql]="agents/gciql.py"
  [gcivl]="agents/gcivl.py"
)

# scene-play-v0: same hyperparameters as cube-double-play
declare -A SCENE_FLAGS=(
  [crl]="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"
  [hiql]="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"
  [qrl]="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"
  [gciql]="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0"
  [gcivl]="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"
)

# puzzle-3x3-play-v0: same hyperparameters as cube/scene
declare -A PUZZLE_FLAGS=(
  [crl]="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"
  [hiql]="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"
  [qrl]="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"
  [gciql]="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0"
  [gcivl]="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"
)

# powderworld-hard-play-v0: same hyperparameters as powderworld-medium
PWD_COMMON="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --video_episodes=0"
declare -A POWDER_FLAGS=(
  [crl]="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
  [hiql]="$PWD_COMMON --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10"
  [qrl]="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
  [gciql]="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
  [gcivl]="$PWD_COMMON --agent.alpha=3.0"
)

submit_dataset_family() {
  local dataset="$1"
  local short="$2"
  local flags_name="$3"
  local mem="$4"
  local time="$5"
  local seed method job_name
  declare -n flags_ref="$flags_name"

  for seed in 0 1 2; do
    for method in crl hiql qrl gciql gcivl; do
      printf -v job_name "pd_%s_s%d_%s" "$short" "$seed" "$method"
      submit_1gpu "$job_name" "$seed" "$dataset" "${AGENTS[$method]}" "${flags_ref[$method]}" "$mem" "$time"
    done
  done
}

echo ""
echo "===== Phase D Single-GPU Submission ====="
echo "  account = $ACCOUNT"
echo "  part    = $PARTITION"
echo "  rungrp  = $RUN_GROUP"
echo "  data    = $DATASET_DIR"
echo "  save    = $SAVE_DIR"
echo ""

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

submit_dataset_family "scene-play-v0" "scene" "SCENE_FLAGS" "32G" "08:00:00"
submit_dataset_family "puzzle-3x3-play-v0" "puzzle" "PUZZLE_FLAGS" "32G" "08:00:00"
submit_dataset_family "powderworld-hard-play-v0" "phard" "POWDER_FLAGS" "64G" "24:00:00"

echo ""
echo "Phase D single-GPU: 45 runs in 45 SLURM jobs"
echo "Monitor with: squeue -u \$USER"
