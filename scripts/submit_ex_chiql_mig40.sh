#!/usr/bin/env bash
# Phase-3c: 3 seeds each on antmaze-teleport-navigate-v0 and antmaze-large-stitch-v0.
# Uses ece567w26_class + gpu_mig40 (A100 MIG 3g.40gb slice) for fast queue.
# Class QoS caps at 1 GPU per job, so we submit 6 separate single-seed sbatchs
# (not the 3-on-1 XLA memory-slice pattern — each seed gets its own MIG slice).
#
# Config knobs baked in (from ex_chiql.py defaults on phase3c-clip):
#   head_expectiles = (0.6, 0.65, 0.7, 0.75, 0.8)
#   grad_clip_norm  = 10.0                           # new in Phase-3c
#   pessimism_beta  = 0.5
#   save_interval   = 100000                         # new in Phase-3c
#
# Usage:
#   bash scripts/submit_ex_chiql_mig40.sh
#
# Overrides:
#   ACCOUNT, PARTITION, WANDB_MODE, SAVE_DIR, DATASET_DIR, VENV_PATH, WALLTIME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ogbench.sbatch"

SCRATCH_ROOT="/scratch/mihalcea_root/mihalcea98/eliotfen"

ACCOUNT="${ACCOUNT:-ece567w26_class}"
PARTITION="${PARTITION:-gpu_mig40}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH_ROOT/ogbench/ex_chiql_phase3c}"
DATASET_DIR="${DATASET_DIR:-$SCRATCH_ROOT/ogbench/data}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv/ogbench}"
WALLTIME="${WALLTIME:-07:55:00}"
# Class QoS caps at 8h; keep a 5-min buffer so sbatch has time to finalize.
TRAIN_STEPS="${TRAIN_STEPS:-600000}"
# 600k is well past Phase-3b's step-400k peak and its step-500-600k degradation
# window — enough to tell whether the grad-clip holds σ and the peak through
# late training. Full 1M would need ~12h on this MIG slice, above the cap.

[ -f "$SBATCH" ] || { echo "Missing SLURM template: $SBATCH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "VENV_PATH not found: $VENV_PATH"; exit 1; }

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

# Shared EXTRA_ARGS for every job.
EXTRA="--eval_episodes=50"
EXTRA="$EXTRA --agent.num_value_heads=5"
EXTRA="$EXTRA --agent.num_subgoal_candidates=16"
EXTRA="$EXTRA --agent.pessimism_beta=0.5"
EXTRA="$EXTRA --agent.high_alpha=3.0 --agent.low_alpha=3.0"
EXTRA="$EXTRA --video_episodes=0"
EXTRA="$EXTRA --save_interval=100000"
EXTRA="$EXTRA --train_steps=$TRAIN_STEPS"

ENVS=(antmaze-teleport-navigate-v0 antmaze-large-stitch-v0)

for ENV_NAME in "${ENVS[@]}"; do
  # Short tag for the job name.
  case "$ENV_NAME" in
    antmaze-teleport-navigate-v0) tag="at" ;;
    antmaze-large-stitch-v0)      tag="as" ;;
    *)                             tag="${ENV_NAME:0:6}" ;;
  esac

  for seed in 0 1 2; do
    exports="SEED=$seed"
    exports="$exports,RUN_GROUP=ex_chiql_phase3c"
    exports="$exports,WANDB_MODE=$WANDB_MODE"
    exports="$exports,DATASET_DIR=$DATASET_DIR"
    exports="$exports,SAVE_DIR=$SAVE_DIR"
    exports="$exports,VENV_PATH=$VENV_PATH"
    exports="$exports,ENV_NAME=$ENV_NAME"
    exports="$exports,AGENT_PATH=agents/ex_chiql.py"
    exports="$exports,EXTRA_ARGS=$EXTRA"

    sbatch \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --job-name="ex_${tag}_s${seed}" \
      --mem=32G \
      --time="$WALLTIME" \
      --export="ALL,$exports" \
      "$SBATCH"
  done
done

echo
echo "Submitted ${#ENVS[@]} envs × 3 seeds = $(( ${#ENVS[@]} * 3 )) jobs (account=$ACCOUNT, partition=$PARTITION, walltime=$WALLTIME)."
echo "Check queue with:  squeue -u \$USER -o '%.10i %.20j %.10T %.10M %.6D %R'"
