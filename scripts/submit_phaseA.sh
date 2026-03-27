#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH_1GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench.sbatch"
SBATCH_2GPU="$REPO_ROOT/cluster/greatlakes/slurm/train_ogbench_2gpu.sbatch"

ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_GROUP="${RUN_GROUP:-phaseA}"
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

COMMON_EXPORT="SEED=$SEED,RUN_GROUP=$RUN_GROUP,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"

submit_2gpu() {
  local job_name="$1"
  local r1_env="$2"; local r1_agent="$3"; local r1_extra="$4"
  local r2_env="$5"; local r2_agent="$6"; local r2_extra="$7"
  local mem="${8:-64G}"
  local time="${9:-24:00:00}"

  echo "Submitting 2-GPU job: $job_name"
  echo "  GPU0: $r1_env + $r1_agent"
  echo "  GPU1: $r2_env + $r2_agent"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$job_name" \
    --mem="$mem" \
    --time="$time" \
    --export="ALL,$COMMON_EXPORT,RUN1_ENV_NAME=$r1_env,RUN1_AGENT_PATH=$r1_agent,RUN1_EXTRA_ARGS=$r1_extra,RUN2_ENV_NAME=$r2_env,RUN2_AGENT_PATH=$r2_agent,RUN2_EXTRA_ARGS=$r2_extra" \
    "$SBATCH_2GPU"
}

submit_1gpu() {
  local job_name="$1"
  local env_name="$2"
  local agent_path="$3"
  local extra_args="$4"
  local mem="${5:-32G}"
  local time="${6:-24:00:00}"

  echo "Submitting 1-GPU job: $job_name ($env_name + $agent_path)"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$job_name" \
    --mem="$mem" \
    --time="$time" \
    --export="ALL,$COMMON_EXPORT,ENV_NAME=$env_name,AGENT_PATH=$agent_path,EXTRA_ARGS=$extra_args" \
    "$SBATCH_1GPU"
}

# =============================================================================
# Phase A: 3 datasets x 5 methods = 15 runs -> 7 x 2-GPU jobs + 1 x 1-GPU job
# =============================================================================

echo ""
echo "===== Phase A: 15 runs packed into 8 SLURM jobs (2 GPUs each) ====="
echo ""

# --- Job 1: antmaze CRL + HIQL ---
submit_2gpu "exp_01" \
  "antmaze-large-navigate-v0" "agents/crl.py"  "--eval_episodes=50 --agent.alpha=0.1 --video_episodes=0" \
  "antmaze-large-navigate-v0" "agents/hiql.py" "--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"

# --- Job 2: antmaze QRL + GCIQL ---
submit_2gpu "exp_02" \
  "antmaze-large-navigate-v0" "agents/qrl.py"   "--eval_episodes=50 --agent.alpha=0.003 --video_episodes=0" \
  "antmaze-large-navigate-v0" "agents/gciql.py" "--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"

# --- Job 3: antmaze GCIVL + cube CRL ---
submit_2gpu "exp_03" \
  "antmaze-large-navigate-v0" "agents/gcivl.py" "--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0" \
  "cube-double-play-v0"       "agents/crl.py"   "--eval_episodes=50 --agent.alpha=3.0 --video_episodes=0"

# --- Job 4: cube HIQL + QRL ---
submit_2gpu "exp_04" \
  "cube-double-play-v0" "agents/hiql.py" "--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 --video_episodes=0" \
  "cube-double-play-v0" "agents/qrl.py"  "--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"

# --- Job 5: cube GCIQL + GCIVL ---
submit_2gpu "exp_05" \
  "cube-double-play-v0" "agents/gciql.py" "--eval_episodes=50 --agent.alpha=1.0 --video_episodes=0" \
  "cube-double-play-v0" "agents/gcivl.py" "--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# --- Powderworld (pixel-based, needs more memory and time) ---
PWD_COMMON="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small --video_episodes=0"

# --- Job 6: powderworld CRL + HIQL ---
submit_2gpu "exp_06" \
  "powderworld-medium-play-v0" "agents/crl.py"  "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "powderworld-medium-play-v0" "agents/hiql.py" "$PWD_COMMON --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10" \
  "96G" "24:00:00"

# --- Job 7: powderworld QRL + GCIQL ---
submit_2gpu "exp_07" \
  "powderworld-medium-play-v0" "agents/qrl.py"   "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "powderworld-medium-play-v0" "agents/gciql.py" "$PWD_COMMON --agent.actor_loss=awr --agent.alpha=3.0" \
  "96G" "24:00:00"

# --- Job 8: powderworld GCIVL (only 1 left, single GPU) ---
submit_1gpu "exp_08" \
  "powderworld-medium-play-v0" "agents/gcivl.py" \
  "$PWD_COMMON --agent.alpha=3.0" \
  "64G" "24:00:00"

echo ""
echo "Phase A submitted: 15 runs in 8 SLURM jobs (seed=$SEED)"
echo "  - 5 state-based 2-GPU jobs (antmaze + cube)"
echo "  - 2 powderworld 2-GPU jobs"
echo "  - 1 powderworld 1-GPU job"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check output: cat ogbench-2gpu-<JOBID>.out"
