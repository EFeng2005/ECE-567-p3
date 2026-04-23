#!/usr/bin/env bash
# Launch 3 EX-HIQL seeds in parallel on the MIG slice we currently hold.
# Intended to be invoked via `ssh gl1251 bash scripts/launch_on_gl1251.sh`.
# Processes fully detach (setsid + nohup) so ssh returns immediately.
set -euo pipefail

REPO="/home/eliotfen/A.ece567/replication/ECE-567-p2"
VENV="$REPO/.venv/ogbench"
OGB="$REPO/external/ogbench/impls"

ENV_NAME="${ENV_NAME:-antmaze-teleport-navigate-v0}"
RUN_GROUP="${RUN_GROUP:-ex_chiql_phase3c}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SCRATCH="/scratch/mihalcea_root/mihalcea98/eliotfen"
SAVE_DIR="${SAVE_DIR:-$SCRATCH/ogbench/ex_chiql_phase3c}"
DATASET_DIR="${DATASET_DIR:-$SCRATCH/ogbench/data}"
LOG_DIR="${LOG_DIR:-$SCRATCH/ogbench/phase3c_logs}"

mkdir -p "$SAVE_DIR" "$DATASET_DIR" "$LOG_DIR"

source "$VENV/bin/activate"
NVIDIA_SITE="$VENV/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_SITE" ]; then
  NVIDIA_LD=$(find "$NVIDIA_SITE" -name lib -type d 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH="${NVIDIA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

cd "$OGB"

tag="${ENV_NAME:0:3}"
stagger=0
for seed in 0 1 2; do
  LOG="$LOG_DIR/${ENV_NAME}_seed${seed}.log"
  echo "launching env=$ENV_NAME seed=$seed log=$LOG (stagger=${stagger}s)"
  setsid nohup bash -c "
    sleep $stagger
    python -u main.py \
      --env_name=$ENV_NAME \
      --agent=agents/ex_chiql.py \
      --seed=$seed \
      --run_group=$RUN_GROUP \
      --save_dir=$SAVE_DIR \
      --dataset_dir=$DATASET_DIR \
      --wandb_mode=disabled \
      --train_steps=$TRAIN_STEPS \
      --log_interval=5000 \
      --eval_interval=100000 \
      --save_interval=100000 \
      --eval_episodes=50 \
      --video_episodes=0 \
      --agent.num_value_heads=5 \
      --agent.num_subgoal_candidates=16 \
      --agent.pessimism_beta=0.5 \
      --agent.high_alpha=3.0 \
      --agent.low_alpha=3.0 \
      >>\"$LOG\" 2>&1
  " </dev/null >/dev/null 2>&1 &
  disown
  stagger=$((stagger + 30))
done

sleep 3
echo "--- processes ---"
ps -o pid,etime,cmd -C python | grep -v grep || echo "(no python processes yet, give it 60s)"
echo
echo "logs in: $LOG_DIR"
echo "checkpoints in: $SAVE_DIR"
