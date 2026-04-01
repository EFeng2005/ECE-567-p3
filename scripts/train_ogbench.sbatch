#!/bin/bash
#SBATCH --job-name=ogbench-train
#SBATCH --output=ogbench-train-%j.out
#SBATCH --account=ece567w26_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OGBENCH_DIR="${OGBENCH_DIR:-$REPO_ROOT/external/ogbench}"
ENV_NAME="${ENV_NAME:-antmaze-large-navigate-v0}"
AGENT_PATH="${AGENT_PATH:-agents/hiql.py}"
SEED="${SEED:-0}"
RUN_GROUP="${RUN_GROUP:-Debug}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

cd "$REPO_ROOT"

# Load modules on Great Lakes
module load python/3.10.4 cuda/12.3.0 2>/dev/null || true

if [ -n "${VENV_PATH:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"

  # Add pip-installed NVIDIA CUDA libs to LD_LIBRARY_PATH so JAX can find them
  NVIDIA_SITE="$VENV_PATH/lib/python3.10/site-packages/nvidia"
  if [ -d "$NVIDIA_SITE" ]; then
    NVIDIA_LD=$(find "$NVIDIA_SITE" -name "lib" -type d 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVIDIA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [ ! -d "$OGBENCH_DIR/impls" ]; then
  echo "Missing upstream OGBench checkout at: $OGBENCH_DIR"
  echo "Run: bash scripts/bootstrap_ogbench.sh"
  exit 1
fi

cd "$OGBENCH_DIR/impls"

echo "Running OGBench training"
echo "  env   = $ENV_NAME"
echo "  agent = $AGENT_PATH"
echo "  seed  = $SEED"
echo "  group = $RUN_GROUP"
echo "  wandb = $WANDB_MODE"
echo "  data  = $DATASET_DIR"
echo "  gl    = $MUJOCO_GL"
echo "  save  = $SAVE_DIR"

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

python_args=(
  --env_name="$ENV_NAME"
  --agent="$AGENT_PATH"
  --seed="$SEED"
  --run_group="$RUN_GROUP"
  --save_dir="$SAVE_DIR"
  --wandb_mode="$WANDB_MODE"
  --dataset_dir="$DATASET_DIR"
)

cmd=(python main.py "${python_args[@]}")
if [ -n "$EXTRA_ARGS" ]; then
  # EXTRA_ARGS is intentionally shell-split so official flag bundles can be passed through sbatch exports.
  # shellcheck disable=SC2206
  extra_args=($EXTRA_ARGS)
  cmd+=("${extra_args[@]}")
fi

printf '  cmd   ='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
