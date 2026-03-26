#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench.sbatch"

ACCOUNT="${ACCOUNT:-ece567w26_class}"
PARTITION="${PARTITION:-spgpu}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-small_matrix}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Missing SLURM template at $SBATCH_SCRIPT"
  exit 1
fi

submit_run() {
  local job_name="$1"
  local env_name="$2"
  local agent_path="$3"
  local extra_args="$4"

  echo "Submitting $job_name"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$job_name" \
    --export="ALL,ENV_NAME=$env_name,AGENT_PATH=$agent_path,SEED=$SEED,RUN_GROUP=${RUN_GROUP_PREFIX}_${job_name},WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,EXTRA_ARGS=$extra_args" \
    "$SBATCH_SCRIPT"
}

# Recommended order from docs/plan/replication_matrix.md:
# 1. antmaze-large-navigate-v0 + HIQL
# 2. cube-double-play-v0 + HIQL
# 3. antmaze-large-navigate-v0 + GCIQL
# 4. cube-double-play-v0 + GCIQL
submit_run "ant_hiql" "antmaze-large-navigate-v0" "agents/hiql.py" "--agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"
submit_run "cube_hiql" "cube-double-play-v0" "agents/hiql.py" "--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"
submit_run "ant_gciql" "antmaze-large-navigate-v0" "agents/gciql.py" "--agent.alpha=0.3 --video_episodes=0"
submit_run "cube_gciql" "cube-double-play-v0" "agents/gciql.py" "--agent.alpha=1.0 --video_episodes=0"
