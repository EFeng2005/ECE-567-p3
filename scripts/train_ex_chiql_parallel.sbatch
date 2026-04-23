#!/bin/bash
#SBATCH --job-name=ex-chiql-parallel
#SBATCH --output=ogbench-train-%j.out
#SBATCH --account=mihalcea98
#SBATCH --partition=spgpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Run 3 EX-HIQL seeds in parallel on a single GPU by giving each JAX
# process a 30% memory slice (mirroring scripts/run_ex_chiql_local.sh).
# All three seeds share the same GPU, so compute is serialized at the
# kernel level but each has its own Python process / optimizer state.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OGBENCH_DIR="${OGBENCH_DIR:-$REPO_ROOT/external/ogbench}"
ENV_NAME="${ENV_NAME:-antmaze-teleport-navigate-v0}"
AGENT_PATH="${AGENT_PATH:-agents/ex_chiql.py}"
RUN_GROUP="${RUN_GROUP:-ex_chiql_phase3b}"
SEEDS="${SEEDS:-0 1 2}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

cd "$REPO_ROOT"

module load python/3.10.4 cuda/12.3.0 2>/dev/null || true

if [ -n "${VENV_PATH:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  NVIDIA_SITE="$VENV_PATH/lib/python3.10/site-packages/nvidia"
  if [ -d "$NVIDIA_SITE" ]; then
    NVIDIA_LD=$(find "$NVIDIA_SITE" -name "lib" -type d 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVIDIA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Share one GPU across 3 processes by disabling preallocation and capping each at 30%.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

if [ ! -d "$OGBENCH_DIR/impls" ]; then
  echo "Missing upstream OGBench checkout at: $OGBENCH_DIR"
  echo "Run: bash scripts/bootstrap_ogbench.sh"
  exit 1
fi

cd "$OGBENCH_DIR/impls"

mkdir -p "$DATASET_DIR" "$SAVE_DIR"
LOG_DIR="$SAVE_DIR/slurm_${SLURM_JOB_ID:-local}_logs"
mkdir -p "$LOG_DIR"

echo "Launching 3 parallel EX-HIQL seeds on $(hostname), GPU $(echo ${CUDA_VISIBLE_DEVICES:-?})"
echo "  env   = $ENV_NAME"
echo "  agent = $AGENT_PATH"
echo "  group = $RUN_GROUP"
echo "  save  = $SAVE_DIR"
echo "  log   = $LOG_DIR"
echo "  seeds = $SEEDS"

pids=()
stagger=0
for SEED in $SEEDS; do
  LOG_FILE="$LOG_DIR/seed${SEED}.log"
  echo "  -> seed=$SEED log=$LOG_FILE"
  (
    # 30s stagger between launches keeps JAX compiles from stampeding.
    sleep "$stagger"
    python -u main.py \
      --env_name="$ENV_NAME" \
      --agent="$AGENT_PATH" \
      --seed="$SEED" \
      --run_group="$RUN_GROUP" \
      --save_dir="$SAVE_DIR" \
      --dataset_dir="$DATASET_DIR" \
      --wandb_mode="$WANDB_MODE" \
      --train_steps="$TRAIN_STEPS" \
      --eval_interval="$EVAL_INTERVAL" \
      --eval_episodes="$EVAL_EPISODES" \
      --save_interval="$SAVE_INTERVAL" \
      --video_episodes=0 \
      --agent.num_value_heads=5 \
      --agent.num_subgoal_candidates=16 \
      --agent.pessimism_beta=0.5 \
      --agent.high_alpha=3.0 \
      --agent.low_alpha=3.0 \
      >>"$LOG_FILE" 2>&1
  ) &
  pids+=($!)
  stagger=$((stagger + 30))
done

echo "Waiting on seeds: ${pids[*]}"
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "seed pid=$pid FAILED"
    rc=1
  fi
done
echo "All seeds finished. exit=$rc"
exit $rc
