#!/usr/bin/env bash
# Phase C: antmaze-large-stitch, antmaze-teleport-navigate, humanoidmaze-medium-navigate
# All state-based. 3 datasets x 5 methods x 3 seeds = 45 runs.
#
# Packing strategy:
#   Seed 0: different experiments paired on 2 GPUs → 8 SLURM jobs
#   Seeds 1+2: same experiment paired (seed1 GPU0, seed2 GPU1) → 15 SLURM jobs
#   Total: 23 SLURM jobs

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
  local run_group="${9:-phaseC}"
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
  local run_group="${5:-phaseC}"
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
  local run_group="${6:-phaseC}"
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

# antmaze-large-stitch: same alphas as navigate, BUT all methods add randomgoal/trajgoal
STITCH_EXTRA="--agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
S_CRL="--eval_episodes=50 --agent.alpha=0.1 $STITCH_EXTRA --video_episodes=0"
S_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 $STITCH_EXTRA --video_episodes=0"
S_QRL="--eval_episodes=50 --agent.alpha=0.003 $STITCH_EXTRA --video_episodes=0"
S_GCIQL="--eval_episodes=50 --agent.alpha=0.3 $STITCH_EXTRA --video_episodes=0"
S_GCIVL="--eval_episodes=50 --agent.alpha=10.0 $STITCH_EXTRA --video_episodes=0"

# antmaze-teleport-navigate: same hyperparameters as antmaze-large-navigate
T_CRL="--eval_episodes=50 --agent.alpha=0.1 --video_episodes=0"
T_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"
T_QRL="--eval_episodes=50 --agent.alpha=0.003 --video_episodes=0"
T_GCIQL="--eval_episodes=50 --agent.alpha=0.3 --video_episodes=0"
T_GCIVL="--eval_episodes=50 --agent.alpha=10.0 --video_episodes=0"

# humanoidmaze-medium-navigate: all methods need discount=0.995
H_CRL="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 --video_episodes=0"
H_HIQL="--eval_episodes=50 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 --video_episodes=0"
H_QRL="--eval_episodes=50 --agent.alpha=0.001 --agent.discount=0.995 --video_episodes=0"
H_GCIQL="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 --video_episodes=0"
H_GCIVL="--eval_episodes=50 --agent.alpha=10.0 --agent.discount=0.995 --video_episodes=0"

STITCH="antmaze-large-stitch-v0"
TELEPORT="antmaze-teleport-navigate-v0"
HMAZE="humanoidmaze-medium-navigate-v0"

# =============================================================================
# PART 1: Seed 0 — 15 runs packed into 8 SLURM jobs
# =============================================================================

echo ""
echo "===== Phase C: seed 0 (15 runs → 8 jobs) ====="

submit_2gpu "expR_01" 0 "$STITCH" "agents/crl.py" "$S_CRL" "$STITCH" "agents/hiql.py" "$S_HIQL"
submit_2gpu "expR_02" 0 "$STITCH" "agents/qrl.py" "$S_QRL" "$STITCH" "agents/gciql.py" "$S_GCIQL"
submit_2gpu "expR_03" 0 "$STITCH" "agents/gcivl.py" "$S_GCIVL" "$TELEPORT" "agents/crl.py" "$T_CRL"
submit_2gpu "expR_04" 0 "$TELEPORT" "agents/hiql.py" "$T_HIQL" "$TELEPORT" "agents/qrl.py" "$T_QRL"
submit_2gpu "expR_05" 0 "$TELEPORT" "agents/gciql.py" "$T_GCIQL" "$TELEPORT" "agents/gcivl.py" "$T_GCIVL"
submit_2gpu "expR_06" 0 "$HMAZE" "agents/crl.py" "$H_CRL" "$HMAZE" "agents/hiql.py" "$H_HIQL"
submit_2gpu "expR_07" 0 "$HMAZE" "agents/qrl.py" "$H_QRL" "$HMAZE" "agents/gciql.py" "$H_GCIQL"
submit_1gpu "expR_08" 0 "$HMAZE" "agents/gcivl.py" "$H_GCIVL"

# =============================================================================
# PART 2: Seeds 1+2 — same experiment paired on 2 GPUs → 15 SLURM jobs
# =============================================================================

echo ""
echo "===== Phase C: seeds 1+2 (30 runs → 15 jobs) ====="

# stitch (5 methods)
submit_2gpu_seeds "expR_09" "$STITCH" "agents/crl.py" "$S_CRL"
submit_2gpu_seeds "expR_10" "$STITCH" "agents/hiql.py" "$S_HIQL"
submit_2gpu_seeds "expR_11" "$STITCH" "agents/qrl.py" "$S_QRL"
submit_2gpu_seeds "expR_12" "$STITCH" "agents/gciql.py" "$S_GCIQL"
submit_2gpu_seeds "expR_13" "$STITCH" "agents/gcivl.py" "$S_GCIVL"

# teleport (5 methods)
submit_2gpu_seeds "expR_14" "$TELEPORT" "agents/crl.py" "$T_CRL"
submit_2gpu_seeds "expR_15" "$TELEPORT" "agents/hiql.py" "$T_HIQL"
submit_2gpu_seeds "expR_16" "$TELEPORT" "agents/qrl.py" "$T_QRL"
submit_2gpu_seeds "expR_17" "$TELEPORT" "agents/gciql.py" "$T_GCIQL"
submit_2gpu_seeds "expR_18" "$TELEPORT" "agents/gcivl.py" "$T_GCIVL"

# humanoidmaze (5 methods)
submit_2gpu_seeds "expR_19" "$HMAZE" "agents/crl.py" "$H_CRL"
submit_2gpu_seeds "expR_20" "$HMAZE" "agents/hiql.py" "$H_HIQL"
submit_2gpu_seeds "expR_21" "$HMAZE" "agents/qrl.py" "$H_QRL"
submit_2gpu_seeds "expR_22" "$HMAZE" "agents/gciql.py" "$H_GCIQL"
submit_2gpu_seeds "expR_23" "$HMAZE" "agents/gcivl.py" "$H_GCIVL"

echo ""
echo "Phase C: 45 runs in 23 SLURM jobs"
echo "  - antmaze-large-stitch-v0: 5 methods x 3 seeds"
echo "  - antmaze-teleport-navigate-v0: 5 methods x 3 seeds"
echo "  - humanoidmaze-medium-navigate-v0: 5 methods x 3 seeds"
echo ""
echo "Monitor with: squeue -u \$USER"
