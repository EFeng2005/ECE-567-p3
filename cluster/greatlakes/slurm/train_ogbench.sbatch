#!/bin/bash
#SBATCH --job-name=ogbench-train
#SBATCH --output=ogbench-train-%j.out
#SBATCH --time=12:00:00
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_ROOT"

# Uncomment and edit these lines to match the environment available on Great Lakes.
# module purge
# module load python
# module load cuda

if [ -n "${VENV_PATH:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

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

python main.py \
  --env_name="$ENV_NAME" \
  --agent="$AGENT_PATH" \
  --seed="$SEED" \
  ${EXTRA_ARGS}

