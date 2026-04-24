#!/usr/bin/env bash
# Wave-based Phase-3b multi-env submission.
#
# Submits one wave of 3 single-GPU jobs at a time, keeping the SLURM queue
# visible to the user at only 3 jobs per wave (low-profile on the lab queue).
#
# Wave 1: pointmaze-teleport-navigate-v0  HIQL     seeds {0,1,2}  -> test1,2,3
# Wave 2: pointmaze-teleport-navigate-v0  EX-HIQL  seeds {0,1,2}  -> test4,5,6
# Wave 3: antmaze-large-navigate-v0        EX-HIQL  seeds {0,1,2}  -> test7,8,9
#
# Usage:
#   bash scripts/submit_phase3b_wave.sh 1
#   # wait for squeue -u $USER to drain, then:
#   bash scripts/submit_phase3b_wave.sh 2
#   bash scripts/submit_phase3b_wave.sh 3
#
# Env overrides: ACCOUNT, PARTITION, VENV_PATH, WANDB_MODE, TIME_LIMIT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ogbench.sbatch"

WAVE="${1:?Usage: $0 <1|2|3>}"

ACCOUNT="${ACCOUNT:-chaijy2}"
PARTITION="${PARTITION:-spgpu}"
WANDB_MODE="${WANDB_MODE:-disabled}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/phase3b_multienv}"
DATASET_DIR="${DATASET_DIR:-$HOME/.ogbench/data}"
VENV_PATH="${VENV_PATH:-$HOME/ece567_venv}"

[ -f "$SBATCH" ]    || { echo "Missing SLURM template: $SBATCH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "Missing venv at: $VENV_PATH"; exit 1; }

case "$WAVE" in
  1)
    ENV_NAME="pointmaze-teleport-navigate-v0"
    AGENT_PATH="agents/hiql.py"
    RUN_GROUP="hiql_ptn_phase3b_multienv"
    EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"
    JOB_OFFSET=0
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
    ;;
  2)
    ENV_NAME="pointmaze-teleport-navigate-v0"
    AGENT_PATH="agents/ex_chiql.py"
    RUN_GROUP="ex_chiql_ptn_phase3b_multienv"
    EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"
    JOB_OFFSET=3
    TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
    ;;
  3)
    ENV_NAME="antmaze-large-navigate-v0"
    AGENT_PATH="agents/ex_chiql.py"
    RUN_GROUP="ex_chiql_aln_phase3b_multienv"
    EXTRA_ARGS="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --video_episodes=0"
    JOB_OFFSET=6
    TIME_LIMIT="${TIME_LIMIT:-10:00:00}"
    ;;
  *)
    echo "Unknown wave: $WAVE. Use 1, 2, or 3."
    exit 1
    ;;
esac

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

echo "===== Wave $WAVE ====="
echo "  env        = $ENV_NAME"
echo "  agent      = $AGENT_PATH"
echo "  run_group  = $RUN_GROUP"
echo "  account    = $ACCOUNT"
echo "  partition  = $PARTITION"
echo "  time limit = $TIME_LIMIT"
echo "  venv       = $VENV_PATH"
echo "  save_dir   = $SAVE_DIR"
echo "  dataset    = $DATASET_DIR"
echo ""

for s in 0 1 2; do
  jobnum=$((JOB_OFFSET + s + 1))
  jobname="test${jobnum}"

  exports="SEED=$s,RUN_GROUP=$RUN_GROUP,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,ENV_NAME=$ENV_NAME,AGENT_PATH=$AGENT_PATH,EXTRA_ARGS=$EXTRA_ARGS"

  echo "[$jobname] seed=$s"
  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$jobname" \
    --output="$REPO_ROOT/slurm-%j.out" \
    --time="$TIME_LIMIT" \
    --export="ALL,$exports" \
    "$SBATCH"
done

echo ""
echo "Wave $WAVE submitted. Monitor: squeue -u \$USER"
