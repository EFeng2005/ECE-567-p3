#!/bin/bash
#SBATCH --job-name=ogbench-2gpu
#SBATCH --output=ogbench-2gpu-%j.out
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# This script runs TWO independent OGBench training jobs on 2 GPUs in parallel.
# Pass the two experiments via RUN1_* and RUN2_* environment variables.
#
# Usage:
#   sbatch --export=ALL,\
#     RUN1_ENV_NAME=antmaze-large-navigate-v0,RUN1_AGENT_PATH=agents/crl.py,RUN1_EXTRA_ARGS="...",\
#     RUN2_ENV_NAME=antmaze-large-navigate-v0,RUN2_AGENT_PATH=agents/hiql.py,RUN2_EXTRA_ARGS="...",\
#     SEED=0,WANDB_MODE=offline,DATASET_DIR=...,SAVE_DIR=...,VENV_PATH=... \
#     cluster/greatlakes/slurm/train_ogbench_2gpu.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OGBENCH_DIR="${OGBENCH_DIR:-$REPO_ROOT/external/ogbench}"
SEED="${SEED:-0}"
RUN_GROUP="${RUN_GROUP:-phaseA}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
elif [ -d "/scratch/chaijy_root/chaijy2" ]; then
  DEFAULT_DATASET_DIR="/scratch/chaijy_root/chaijy2/$USER/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

cd "$REPO_ROOT"

module load python/3.10.4 cuda/12.3.0 2>/dev/null || true

if [ -n "${VENV_PATH:-}" ]; then
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
  exit 1
fi

cd "$OGBENCH_DIR/impls"
mkdir -p "$DATASET_DIR" "$SAVE_DIR"

run_experiment() {
  local gpu_id="$1"
  local env_name="$2"
  local agent_path="$3"
  local extra_args="$4"
  local run_label="$5"
  local run_seed="$6"

  echo "[GPU $gpu_id] Starting: $env_name + $agent_path (seed=$run_seed)"

  # shellcheck disable=SC2206
  local extra=($extra_args)

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --env_name="$env_name" \
    --agent="$agent_path" \
    --seed="$run_seed" \
    --run_group="${RUN_GROUP}_${run_label}" \
    --save_dir="$SAVE_DIR" \
    --wandb_mode="$WANDB_MODE" \
    --dataset_dir="$DATASET_DIR" \
    "${extra[@]}"

  echo "[GPU $gpu_id] Finished: $env_name + $agent_path"
}

# Per-run seeds: fall back to global SEED if not set
RUN1_SEED="${RUN1_SEED:-$SEED}"
RUN2_SEED="${RUN2_SEED:-$SEED}"

# Run two experiments in parallel, one per GPU
run_experiment 0 "${RUN1_ENV_NAME}" "${RUN1_AGENT_PATH}" "${RUN1_EXTRA_ARGS:-}" "run1" "$RUN1_SEED" &
PID1=$!

run_experiment 1 "${RUN2_ENV_NAME}" "${RUN2_AGENT_PATH}" "${RUN2_EXTRA_ARGS:-}" "run2" "$RUN2_SEED" &
PID2=$!

echo "Waiting for both experiments to finish..."
echo "  PID1=$PID1 (GPU 0): ${RUN1_ENV_NAME} + ${RUN1_AGENT_PATH} (seed=$RUN1_SEED)"
echo "  PID2=$PID2 (GPU 1): ${RUN2_ENV_NAME} + ${RUN2_AGENT_PATH} (seed=$RUN2_SEED)"

# Wait for both and capture exit codes
FAIL=0
wait $PID1 || FAIL=1
wait $PID2 || FAIL=1

if [ $FAIL -ne 0 ]; then
  echo "WARNING: One or both experiments failed. Check output above."
  exit 1
fi

echo "Both experiments completed successfully."
