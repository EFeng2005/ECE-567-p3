#!/usr/bin/env bash
# Single-seed EX-HIQL-clip run with configurable head_expectiles.
#
# Usage: bash scripts/run_wide_tau_clip_local.sh <seed> <tau_tag>
#   tau_tag ∈ { wide | mid }
#     wide  → head_expectiles=(0.1, 0.3, 0.5, 0.7, 0.9), save to ex_chiql_wide_clip
#     mid   → head_expectiles=(0.4, 0.5, 0.6, 0.7, 0.8), save to ex_chiql_mid_clip
#
# Both use indep-trunk EX-HIQL + optax.clip_by_global_norm(10.0). Lives on
# the `wide-tau-clip` branch; uses a separate agent filename (ex_chiql_clip.py)
# so it doesn't fight for agents/ex_chiql.py in ogbench_full with any other
# concurrent shared-trunk EX-HIQL run.
set -euo pipefail

SEED="${1:?usage: run_wide_tau_clip_local.sh <seed> <tau_tag>}"
TAG="${2:?usage: run_wide_tau_clip_local.sh <seed> <tau_tag>}"

case "$TAG" in
  wide)
    HEAD_EXPECTILES="(0.1, 0.3, 0.5, 0.7, 0.9)"
    RUN_GROUP="ex_chiql_wide_clip"
    ;;
  mid)
    HEAD_EXPECTILES="(0.4, 0.5, 0.6, 0.7, 0.8)"
    RUN_GROUP="ex_chiql_mid_clip"
    ;;
  *)
    echo "tau_tag must be 'wide' or 'mid'" >&2; exit 1 ;;
esac

REPO="/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567"
VENV="$HOME/ece567/.venv"
OGB="$REPO/external/ogbench_full/impls"

# Sync the tracked ex_chiql_clip.py into ogbench_full (separate filename from
# ex_chiql.py, so this doesn't interfere with any other concurrent run).
cp "$REPO/external/ogbench/impls/agents/ex_chiql_clip.py" "$OGB/agents/ex_chiql_clip.py"

export OGBENCH_DATASET_DIR="$HOME/ece567/data"
SAVE_DIR="$HOME/ece567/runs/$RUN_GROUP"
LOG_DIR="$HOME/ece567/logs"
mkdir -p "$OGBENCH_DATASET_DIR" "$SAVE_DIR" "$LOG_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Lower per-process cap so 9 concurrent procs (3 current + 3 wide + 3 mid) fit in 24 GB.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

cd "$OGB"

LOG_FILE="$LOG_DIR/${RUN_GROUP}_seed${SEED}.log"
echo "Starting $RUN_GROUP seed=$SEED head_expectiles=$HEAD_EXPECTILES. Log: $LOG_FILE"

"$VENV/bin/python" -u main.py \
  --env_name=antmaze-teleport-navigate-v0 \
  --agent=agents/ex_chiql_clip.py \
  --seed="$SEED" \
  --run_group="$RUN_GROUP" \
  --save_dir="$SAVE_DIR" \
  --dataset_dir="$OGBENCH_DATASET_DIR" \
  --wandb_mode=disabled \
  --train_steps=1000000 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --save_interval=1000000 \
  --eval_episodes=50 \
  --video_episodes=0 \
  --agent.num_value_heads=5 \
  --agent.head_expectiles="$HEAD_EXPECTILES" \
  --agent.num_subgoal_candidates=16 \
  --agent.pessimism_beta=0.5 \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  --agent.grad_clip=10.0 \
  >>"$LOG_FILE" 2>&1
