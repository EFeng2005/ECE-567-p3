#!/usr/bin/env bash
# Phase B: Same 3 datasets x 5 methods as Phase A, but seeds 1 and 2.
# Each 2-GPU job pairs seed=1 (GPU0) and seed=2 (GPU1) for the same experiment.
# Total: 30 runs in 15 SLURM jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_2GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench_2gpu.sbatch"

ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_GROUP="${RUN_GROUP:-phaseB}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
elif [ -d "/scratch/chaijy_root/chaijy2" ]; then
  DEFAULT_DATASET_DIR="/scratch/chaijy_root/chaijy2/$USER/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

VENV_PATH="${VENV_PATH:-}"

if [ ! -f "$SBATCH_2GPU" ]; then
  echo "Missing SLURM template: $SBATCH_2GPU"
  exit 1
fi

# SEED is unused for Phase B (per-run seeds are RUN1_SEED=1, RUN2_SEED=2)
COMMON_EXPORT="SEED=0,RUN_GROUP=$RUN_GROUP,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH,RUN1_SEED=1,RUN2_SEED=2"

submit_2gpu() {
  local job_name="$1"
  local env_name="$2"
  local agent_path="$3"
  local extra_args="$4"
  local mem="${5:-64G}"
  local time="${6:-24:00:00}"

  echo "Submitting 2-GPU job: $job_name ($env_name + $agent_path, seeds 1+2)"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$job_name" \
    --mem="$mem" \
    --time="$time" \
    --export="ALL,$COMMON_EXPORT,RUN1_ENV_NAME=$env_name,RUN1_AGENT_PATH=$agent_path,RUN1_EXTRA_ARGS=$extra_args,RUN2_ENV_NAME=$env_name,RUN2_AGENT_PATH=$agent_path,RUN2_EXTRA_ARGS=$extra_args" \
    "$SBATCH_2GPU"
}

# =============================================================================
# Phase B: 3 datasets x 5 methods x 2 seeds = 30 runs -> 15 x 2-GPU jobs
# Each job: same experiment, seed=1 on GPU0, seed=2 on GPU1
# =============================================================================

echo ""
echo "===== Phase B: 30 runs in 15 SLURM jobs (seed 1+2 per job) ====="
echo ""

# --- antmaze-large-navigate-v0 (5 methods) ---
submit_2gpu "expB_01" \
  "antmaze-large-navigate-v0" "agents/crl.py" \
  "--eval_episodes=50 --agent.alpha=0.1 --video_episodes=0"

submit_2gpu "expB_02" \
  "antmaze-large-navigate-v0" "agents/hiql.py" \
  "--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"

submit_2gpu "expB_03" \
  "antmaze-large-navigate-v0" "agents/qrl.py" \
  "--eval_episodes=50 --agent.alpha=0.003 --video_episodes=0"

submit_2gpu "expB_04" \
  "antmaze-large-navigate-v0" "agents/gciql.py" \
  "--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"

submit_2gpu "expB_05" \
  "antmaze-large-navigate-v0" "agents/gcivl.py" \
  "--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# --- cube-double-play-v0 (5 methods) ---
submit_2gpu "expB_06" \
  "cube-double-play-v0" "agents/crl.py" \
  "--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"

submit_2gpu "expB_07" \
  "cube-double-play-v0" "agents/hiql.py" \
  "--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"

submit_2gpu "expB_08" \
  "cube-double-play-v0" "agents/qrl.py" \
  "--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"

submit_2gpu "expB_09" \
  "cube-double-play-v0" "agents/gciql.py" \
  "--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0"

submit_2gpu "expB_10" \
  "cube-double-play-v0" "agents/gcivl.py" \
  "--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# --- powderworld-medium-play-v0 (5 methods, pixel-based) ---
PWD_COMMON="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --video_episodes=0"

submit_2gpu "expB_11" \
  "powderworld-medium-play-v0" "agents/crl.py" \
  "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "96G" "24:00:00"

submit_2gpu "expB_12" \
  "powderworld-medium-play-v0" "agents/hiql.py" \
  "$PWD_COMMON --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10" \
  "96G" "24:00:00"

submit_2gpu "expB_13" \
  "powderworld-medium-play-v0" "agents/qrl.py" \
  "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "96G" "24:00:00"

submit_2gpu "expB_14" \
  "powderworld-medium-play-v0" "agents/gciql.py" \
  "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "96G" "24:00:00"

submit_2gpu "expB_15" \
  "powderworld-medium-play-v0" "agents/gcivl.py" \
  "$PWD_COMMON --agent.alpha=3.0" \
  "96G" "24:00:00"

echo ""
echo "Phase B submitted: 30 runs in 15 SLURM jobs (seeds 1+2)"
echo "  - 10 state-based jobs (antmaze + cube)"
echo "  - 5 powderworld jobs"
echo ""
echo "Monitor with: squeue -u \$USER"
