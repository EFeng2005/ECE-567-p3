#!/usr/bin/env bash
# Submit Phase-3b EX-HIQL: 3 seeds packed onto a single A40 via JAX memory slicing.
# Tight head_expectiles=(0.6, 0.65, 0.7, 0.75, 0.8) + grad_clip_norm=10.0 are
# baked into ex_chiql.py's get_config() defaults (see EXPECTILE_HIQL.md §6,
# PHASE3_DESIGN_A_REPORT.md §7.1/§7.2).
#
# Usage:
#   bash scripts/submit_ex_chiql.sh
#
# Overrides:
#   ACCOUNT     - SLURM account     (default: mihalcea98)
#   PARTITION   - SLURM partition   (default: spgpu)
#   WANDB_MODE  - wandb mode        (default: offline)
#   SAVE_DIR    - output root       (default: /scratch/mihalcea_root/mihalcea98/eliotfen/ogbench/ex_chiql_runs)
#   DATASET_DIR - dataset cache     (default: /scratch/mihalcea_root/mihalcea98/eliotfen/ogbench/data)
#   VENV_PATH   - python venv root  (default: repo_root/.venv/ogbench)
#   WALLTIME    - sbatch --time     (default: 12:00:00)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ex_chiql_parallel.sbatch"

SCRATCH_ROOT="/scratch/mihalcea_root/mihalcea98/eliotfen"

ACCOUNT="${ACCOUNT:-mihalcea98}"
PARTITION="${PARTITION:-spgpu}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$SCRATCH_ROOT/ogbench/ex_chiql_runs}"
DATASET_DIR="${DATASET_DIR:-$SCRATCH_ROOT/ogbench/data}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv/ogbench}"
WALLTIME="${WALLTIME:-12:00:00}"

[ -f "$SBATCH" ] || { echo "Missing SLURM template: $SBATCH"; exit 1; }
[ -d "$VENV_PATH" ] || { echo "VENV_PATH not found: $VENV_PATH"; exit 1; }

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

exports="REPO_ROOT=$REPO_ROOT"
exports="$exports,RUN_GROUP=ex_chiql_phase3b"
exports="$exports,WANDB_MODE=$WANDB_MODE"
exports="$exports,DATASET_DIR=$DATASET_DIR"
exports="$exports,SAVE_DIR=$SAVE_DIR"
exports="$exports,VENV_PATH=$VENV_PATH"
exports="$exports,ENV_NAME=antmaze-teleport-navigate-v0"
exports="$exports,AGENT_PATH=agents/ex_chiql.py"
exports="$exports,SEEDS=0 1 2"

sbatch \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --job-name="ex_chiql_phase3b" \
  --time="$WALLTIME" \
  --export="ALL,$exports" \
  "$SBATCH"

echo
echo "Submitted 1 sbatch with 3 parallel EX-HIQL seeds (account=$ACCOUNT, partition=$PARTITION, walltime=$WALLTIME)."
echo "Check queue with:  squeue -u \$USER -o '%.10i %.20j %.10T %.10M %.6D %R'"
