#!/usr/bin/env bash
# Phase D: scene-play, puzzle-3x3-play, powderworld-hard-play
# 3 datasets x 5 methods x 3 seeds = 45 runs.
#
# Packing strategy:
#   Seed 0: different experiments paired on 2 GPUs → 8 SLURM jobs
#   Seeds 1+2: same experiment paired (seed1 GPU0, seed2 GPU1) → 15 SLURM jobs
#   Total: 23 SLURM jobs
#
# Usage:
#   export VENV_PATH="/scratch/chaijy_root/chaijy2/$USER/ogbench_venv"
#   export REPO_ROOT="$HOME/projects/ECE-567-p3"
#   bash scripts/submit_phaseD.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_1GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench.sbatch"
SBATCH_2GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench_2gpu.sbatch"

ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
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

for f in "$SBATCH_1GPU" "$SBATCH_2GPU"; do
  if [ ! -f "$f" ]; then
    echo "Missing SLURM template: $f"
    exit 1
  fi
done

# --- Helper: submit 2-GPU job with shared seed ---
submit_2gpu() {
  local job_name="$1"
  local seed="$2"
  local r1_env="$3"; local r1_agent="$4"; local r1_extra="$5"
  local r2_env="$6"; local r2_agent="$7"; local r2_extra="$8"
  local run_group="${9:-phaseD}"
  local mem="${10:-64G}"
  local time="${11:-24:00:00}"

  local exports="SEED=$seed,RUN_GROUP=$run_group,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,RUN1_ENV_NAME=$r1_env,RUN1_AGENT_PATH=$r1_agent,RUN1_EXTRA_ARGS=$r1_extra"
  exports="$exports,RUN2_ENV_NAME=$r2_env,RUN2_AGENT_PATH=$r2_agent,RUN2_EXTRA_ARGS=$r2_extra"

  echo "  $job_name: $r1_env+$(basename $r1_agent .py) | $r2_env+$(basename $r2_agent .py) [seed=$seed]"
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name="$job_name" \
    --mem="$mem" --time="$time" --export="ALL,$exports" "$SBATCH_2GPU"
}

# --- Helper: submit 2-GPU job with per-GPU seeds ---
submit_2gpu_seeds() {
  local job_name="$1"
  local env_name="$2"; local agent_path="$3"; local extra_args="$4"
  local run_group="${5:-phaseD}"
  local mem="${6:-64G}"
  local time="${7:-24:00:00}"

  local exports="SEED=0,RUN_GROUP=$run_group,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,RUN1_SEED=1,RUN2_SEED=2"
  exports="$exports,RUN1_ENV_NAME=$env_name,RUN1_AGENT_PATH=$agent_path,RUN1_EXTRA_ARGS=$extra_args"
  exports="$exports,RUN2_ENV_NAME=$env_name,RUN2_AGENT_PATH=$agent_path,RUN2_EXTRA_ARGS=$extra_args"

  echo "  $job_name: $env_name+$(basename $agent_path .py) [seeds 1+2]"
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name="$job_name" \
    --mem="$mem" --time="$time" --export="ALL,$exports" "$SBATCH_2GPU"
}

# --- Helper: submit 1-GPU job ---
submit_1gpu() {
  local job_name="$1"
  local seed="$2"
  local env_name="$3"; local agent_path="$4"; local extra_args="$5"
  local run_group="${6:-phaseD}"
  local mem="${7:-32G}"
  local time="${8:-24:00:00}"

  local exports="SEED=$seed,RUN_GROUP=$run_group,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,ENV_NAME=$env_name,AGENT_PATH=$agent_path,EXTRA_ARGS=$extra_args"

  echo "  $job_name: $env_name+$(basename $agent_path .py) [seed=$seed]"
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name="$job_name" \
    --mem="$mem" --time="$time" --export="ALL,$exports" "$SBATCH_1GPU"
}

# =============================================================================
# Hyperparameter shorthands
# =============================================================================

# scene-play-v0: same hyperparameters as cube-double-play
SC_CRL="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"
SC_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"
SC_QRL="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"
SC_GCIQL="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0"
SC_GCIVL="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# puzzle-3x3-play-v0: same hyperparameters as cube/scene
PZ_CRL="--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"
PZ_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0"
PZ_QRL="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"
PZ_GCIQL="--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0"
PZ_GCIVL="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# powderworld-hard-play-v0: same hyperparameters as powderworld-medium
PWD_COMMON="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --video_episodes=0"
PH_CRL="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
PH_HIQL="$PWD_COMMON --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10"
PH_QRL="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
PH_GCIQL="$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0"
PH_GCIVL="$PWD_COMMON --agent.alpha=3.0"

SCENE="scene-play-v0"
PUZZLE="puzzle-3x3-play-v0"
PHARD="powderworld-hard-play-v0"

# =============================================================================
# PART 1: Seed 0 — 15 runs packed into 8 SLURM jobs
# =============================================================================

echo ""
echo "===== Phase D: seed 0 (15 runs → 8 jobs) ====="

# scene + puzzle (state-based, 64G)
submit_2gpu "expT_01" 0 "$SCENE" "agents/crl.py" "$SC_CRL" "$SCENE" "agents/hiql.py" "$SC_HIQL"
submit_2gpu "expT_02" 0 "$SCENE" "agents/qrl.py" "$SC_QRL" "$SCENE" "agents/gciql.py" "$SC_GCIQL"
submit_2gpu "expT_03" 0 "$SCENE" "agents/gcivl.py" "$SC_GCIVL" "$PUZZLE" "agents/crl.py" "$PZ_CRL"
submit_2gpu "expT_04" 0 "$PUZZLE" "agents/hiql.py" "$PZ_HIQL" "$PUZZLE" "agents/qrl.py" "$PZ_QRL"
submit_2gpu "expT_05" 0 "$PUZZLE" "agents/gciql.py" "$PZ_GCIQL" "$PUZZLE" "agents/gcivl.py" "$PZ_GCIVL"

# powderworld-hard (pixel-based, 96G)
submit_2gpu "expT_06" 0 "$PHARD" "agents/crl.py" "$PH_CRL" "$PHARD" "agents/hiql.py" "$PH_HIQL" "phaseD" "96G" "24:00:00"
submit_2gpu "expT_07" 0 "$PHARD" "agents/qrl.py" "$PH_QRL" "$PHARD" "agents/gciql.py" "$PH_GCIQL" "phaseD" "96G" "24:00:00"
submit_1gpu "expT_08" 0 "$PHARD" "agents/gcivl.py" "$PH_GCIVL" "phaseD" "64G" "24:00:00"

# =============================================================================
# PART 2: Seeds 1+2 — same experiment paired on 2 GPUs → 15 SLURM jobs
# =============================================================================

echo ""
echo "===== Phase D: seeds 1+2 (30 runs → 15 jobs) ====="

# scene (5 methods)
submit_2gpu_seeds "expT_09" "$SCENE" "agents/crl.py" "$SC_CRL"
submit_2gpu_seeds "expT_10" "$SCENE" "agents/hiql.py" "$SC_HIQL"
submit_2gpu_seeds "expT_11" "$SCENE" "agents/qrl.py" "$SC_QRL"
submit_2gpu_seeds "expT_12" "$SCENE" "agents/gciql.py" "$SC_GCIQL"
submit_2gpu_seeds "expT_13" "$SCENE" "agents/gcivl.py" "$SC_GCIVL"

# puzzle (5 methods)
submit_2gpu_seeds "expT_14" "$PUZZLE" "agents/crl.py" "$PZ_CRL"
submit_2gpu_seeds "expT_15" "$PUZZLE" "agents/hiql.py" "$PZ_HIQL"
submit_2gpu_seeds "expT_16" "$PUZZLE" "agents/qrl.py" "$PZ_QRL"
submit_2gpu_seeds "expT_17" "$PUZZLE" "agents/gciql.py" "$PZ_GCIQL"
submit_2gpu_seeds "expT_18" "$PUZZLE" "agents/gcivl.py" "$PZ_GCIVL"

# powderworld-hard (5 methods, 96G)
submit_2gpu_seeds "expT_19" "$PHARD" "agents/crl.py" "$PH_CRL" "phaseD" "96G" "24:00:00"
submit_2gpu_seeds "expT_20" "$PHARD" "agents/hiql.py" "$PH_HIQL" "phaseD" "96G" "24:00:00"
submit_2gpu_seeds "expT_21" "$PHARD" "agents/qrl.py" "$PH_QRL" "phaseD" "96G" "24:00:00"
submit_2gpu_seeds "expT_22" "$PHARD" "agents/gciql.py" "$PH_GCIQL" "phaseD" "96G" "24:00:00"
submit_2gpu_seeds "expT_23" "$PHARD" "agents/gcivl.py" "$PH_GCIVL" "phaseD" "96G" "24:00:00"

echo ""
echo "Phase D: 45 runs in 23 SLURM jobs"
echo "  - scene-play-v0: 5 methods x 3 seeds"
echo "  - puzzle-3x3-play-v0: 5 methods x 3 seeds"
echo "  - powderworld-hard-play-v0: 5 methods x 3 seeds"
echo ""
echo "Monitor with: squeue -u \$USER"
