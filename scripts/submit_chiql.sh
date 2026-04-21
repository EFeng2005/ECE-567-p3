#!/usr/bin/env bash
# Submit 3 C-HIQL training seeds on antmaze-teleport-navigate-v0.
#
# Defaults target the mihalcea98 SLURM account (our own allocation) and the
# gpu-rtx6000 partition (RTX PRO 6000 Blackwell). Saves artifacts under our
# scratch dir so we do not consume class-account storage.
#
# Usage:
#   bash scripts/submit_chiql.sh
#
# Overrides via env vars (all optional):
#   ACCOUNT       - SLURM account           (default: mihalcea98)
#   PARTITION     - SLURM partition         (default: gpu-rtx6000)
#   WANDB_MODE    - wandb mode              (default: offline)
#   SAVE_DIR      - output root             (default: /scratch/mihalcea_root/mihalcea98/eliotfen/ogbench/chiql_runs)
#   DATASET_DIR   - OGBench dataset cache   (default: /scratch/mihalcea_root/mihalcea98/eliotfen/ogbench/data)
#   VENV_PATH     - python venv root        (default: repo_root/.venv/ogbench)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ogbench.sbatch"

SCRATCH_ROOT="/scratch/mihalcea_root/mihalcea98/eliotfen"

ACCOUNT="${ACCOUNT:-mihalcea98}"
PARTITION="${PARTITION:-gpu-rtx6000}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH_ROOT/ogbench/chiql_runs}"
DATASET_DIR="${DATASET_DIR:-$SCRATCH_ROOT/ogbench/data}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv/ogbench}"

[ -f "$SBATCH" ] || { echo "Missing SLURM template: $SBATCH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "VENV_PATH not found: $VENV_PATH"; exit 1; }

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

# HIQL hyperparameters for the antmaze-teleport/navigate setting (alphas=3.0
# matches the Phase-1 HIQL runs) plus C-HIQL knobs. pessimism_beta=0.5 here
# is inference-only and the checkpoint is beta-independent; the eval sweep
# script will overwrite it with {0.0, 0.25, 0.5, 1.0, 2.0}.
EXTRA="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.num_value_heads=5 --agent.num_subgoal_candidates=16 --agent.pessimism_beta=0.5 --video_episodes=0 --save_interval=1000000"

for seed in 0 1 2; do
  exports="SEED=$seed"
  exports="$exports,RUN_GROUP=chiql_phase2"
  exports="$exports,WANDB_MODE=$WANDB_MODE"
  exports="$exports,DATASET_DIR=$DATASET_DIR"
  exports="$exports,SAVE_DIR=$SAVE_DIR"
  exports="$exports,VENV_PATH=$VENV_PATH"
  exports="$exports,ENV_NAME=antmaze-teleport-navigate-v0"
  exports="$exports,AGENT_PATH=agents/chiql.py"
  exports="$exports,EXTRA_ARGS=$EXTRA"

  sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="chiql_at_s${seed}" \
    --mem=32G \
    --time=08:00:00 \
    --export="ALL,$exports" \
    "$SBATCH"
done

echo
echo "Submitted 3 C-HIQL training jobs (account=$ACCOUNT, partition=$PARTITION)."
echo "Check queue with:   squeue -u \$USER -o '%.10i %.20j %.10T %.10M %.6D %R'"
