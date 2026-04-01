#!/usr/bin/env bash
# Submit all 135 OGBench training runs (9 datasets x 5 methods x 3 seeds).
# Each run uses a single GPU via SLURM.
#
# Usage:
#   export VENV_PATH="/scratch/.../ogbench_venv"
#   bash scripts/submit_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SBATCH="$REPO_ROOT/scripts/train_ogbench.sbatch"

ACCOUNT="${ACCOUNT:-ece567w26_class}"
PARTITION="${PARTITION:-spgpu}"
WANDB_MODE="${WANDB_MODE:-offline}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"

DEFAULT_DATASET_DIR="$HOME/.ogbench/data"
if [ -n "${SCRATCH:-}" ]; then
  DEFAULT_DATASET_DIR="$SCRATCH/ogbench/data"
fi
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
VENV_PATH="${VENV_PATH:-}"

[ -f "$SBATCH" ] || { echo "Missing SLURM template: $SBATCH"; exit 1; }

submit() {
  local job_name="$1" seed="$2" env_name="$3" agent_path="$4" extra_args="$5"
  local mem="${6:-32G}" time="${7:-08:00:00}"

  local exports="SEED=$seed,RUN_GROUP=exp,WANDB_MODE=$WANDB_MODE,DATASET_DIR=$DATASET_DIR,SAVE_DIR=$SAVE_DIR,VENV_PATH=$VENV_PATH"
  exports="$exports,ENV_NAME=$env_name,AGENT_PATH=$agent_path,EXTRA_ARGS=$extra_args"

  sbatch --account="$ACCOUNT" --partition="$PARTITION" --job-name="$job_name" \
    --mem="$mem" --time="$time" --export="ALL,$exports" "$SBATCH"
}

submit_dataset() {
  local dataset="$1" short="$2" mem="$3" time="$4"
  shift 4
  # Remaining args: method agent_path extra_args (repeated for each method)
  local methods=() agents=() extras=()
  while [ $# -gt 0 ]; do
    methods+=("$1"); agents+=("$2"); extras+=("$3"); shift 3
  done

  echo "--- $dataset ---"
  for seed in 0 1 2; do
    for i in "${!methods[@]}"; do
      submit "${short}_s${seed}_${methods[$i]}" "$seed" "$dataset" "${agents[$i]}" "${extras[$i]}" "$mem" "$time"
    done
  done
}

# =============================================================================
# Hyperparameters (from official OGBench hyperparameters.sh)
# =============================================================================

V="--video_episodes=0"

# --- Locomotion: antmaze-large-navigate-v0 ---
AN_CRL="--eval_episodes=50 --agent.alpha=0.1 $V"
AN_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 $V"
AN_QRL="--eval_episodes=50 --agent.alpha=0.003 $V"
AN_GCIQL="--eval_episodes=50 --agent.alpha=0.3 $V"
AN_GCIVL="--eval_episodes=50 --agent.alpha=10.0 $V"

# --- Locomotion: antmaze-large-stitch-v0 ---
STITCH="--agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
AS_CRL="--eval_episodes=50 --agent.alpha=0.1 $STITCH $V"
AS_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 $STITCH $V"
AS_QRL="--eval_episodes=50 --agent.alpha=0.003 $STITCH $V"
AS_GCIQL="--eval_episodes=50 --agent.alpha=0.3 $STITCH $V"
AS_GCIVL="--eval_episodes=50 --agent.alpha=10.0 $STITCH $V"

# --- Locomotion: antmaze-teleport-navigate-v0 (same as navigate) ---
AT_CRL="$AN_CRL"; AT_HIQL="$AN_HIQL"; AT_QRL="$AN_QRL"; AT_GCIQL="$AN_GCIQL"; AT_GCIVL="$AN_GCIVL"

# --- Locomotion: humanoidmaze-medium-navigate-v0 (discount=0.995) ---
HM_CRL="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 $V"
HM_HIQL="--eval_episodes=50 --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 $V"
HM_QRL="--eval_episodes=50 --agent.alpha=0.001 --agent.discount=0.995 $V"
HM_GCIQL="--eval_episodes=50 --agent.alpha=0.1 --agent.discount=0.995 $V"
HM_GCIVL="--eval_episodes=50 --agent.alpha=10.0 --agent.discount=0.995 $V"

# --- Manipulation: cube-double / scene / puzzle (shared alphas) ---
M_CRL="--eval_episodes=50 --agent.alpha=3.0 $V"
M_HIQL="--eval_episodes=50 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10 $V"
M_QRL="--eval_episodes=50 --agent.alpha=0.3 $V"
M_GCIQL="--eval_episodes=50 --agent.alpha=1.0 $V"
M_GCIVL="--eval_episodes=50 --agent.alpha=10.0 $V"

# --- Powderworld: medium / hard (pixel-based) ---
PW="--train_steps=500000 --eval_episodes=50 --eval_temperature=0.3 --eval_on_cpu=0 --agent.batch_size=256 --agent.discrete=True --agent.encoder=impala_small $V"
PW_CRL="$PW --agent.actor_loss=awr --agent.alpha=3.0"
PW_HIQL="$PW --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 --agent.subgoal_steps=10"
PW_QRL="$PW --agent.actor_loss=awr --agent.alpha=3.0"
PW_GCIQL="$PW --agent.actor_loss=awr --agent.alpha=3.0"
PW_GCIVL="$PW --agent.alpha=3.0"

# =============================================================================
# Submit all runs
# =============================================================================

echo ""
echo "===== Submitting all 135 runs (single GPU each) ====="
echo "  account=$ACCOUNT  partition=$PARTITION  wandb=$WANDB_MODE"
echo ""

mkdir -p "$DATASET_DIR" "$SAVE_DIR"

# Locomotion
submit_dataset "antmaze-large-navigate-v0" "an" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$AN_CRL"   "hiql" "agents/hiql.py" "$AN_HIQL" \
  "qrl" "agents/qrl.py" "$AN_QRL"   "gciql" "agents/gciql.py" "$AN_GCIQL" \
  "gcivl" "agents/gcivl.py" "$AN_GCIVL"

submit_dataset "antmaze-large-stitch-v0" "as" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$AS_CRL"   "hiql" "agents/hiql.py" "$AS_HIQL" \
  "qrl" "agents/qrl.py" "$AS_QRL"   "gciql" "agents/gciql.py" "$AS_GCIQL" \
  "gcivl" "agents/gcivl.py" "$AS_GCIVL"

submit_dataset "antmaze-teleport-navigate-v0" "at" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$AT_CRL"   "hiql" "agents/hiql.py" "$AT_HIQL" \
  "qrl" "agents/qrl.py" "$AT_QRL"   "gciql" "agents/gciql.py" "$AT_GCIQL" \
  "gcivl" "agents/gcivl.py" "$AT_GCIVL"

submit_dataset "humanoidmaze-medium-navigate-v0" "hm" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$HM_CRL"   "hiql" "agents/hiql.py" "$HM_HIQL" \
  "qrl" "agents/qrl.py" "$HM_QRL"   "gciql" "agents/gciql.py" "$HM_GCIQL" \
  "gcivl" "agents/gcivl.py" "$HM_GCIVL"

# Manipulation
submit_dataset "cube-double-play-v0" "cb" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$M_CRL"   "hiql" "agents/hiql.py" "$M_HIQL" \
  "qrl" "agents/qrl.py" "$M_QRL"   "gciql" "agents/gciql.py" "$M_GCIQL" \
  "gcivl" "agents/gcivl.py" "$M_GCIVL"

submit_dataset "scene-play-v0" "sc" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$M_CRL"   "hiql" "agents/hiql.py" "$M_HIQL" \
  "qrl" "agents/qrl.py" "$M_QRL"   "gciql" "agents/gciql.py" "$M_GCIQL" \
  "gcivl" "agents/gcivl.py" "$M_GCIVL"

submit_dataset "puzzle-3x3-play-v0" "pz" "32G" "08:00:00" \
  "crl" "agents/crl.py" "$M_CRL"   "hiql" "agents/hiql.py" "$M_HIQL" \
  "qrl" "agents/qrl.py" "$M_QRL"   "gciql" "agents/gciql.py" "$M_GCIQL" \
  "gcivl" "agents/gcivl.py" "$M_GCIVL"

# Powderworld (pixel-based, needs more memory)
submit_dataset "powderworld-medium-play-v0" "pm" "64G" "08:00:00" \
  "crl" "agents/crl.py" "$PW_CRL"   "hiql" "agents/hiql.py" "$PW_HIQL" \
  "qrl" "agents/qrl.py" "$PW_QRL"   "gciql" "agents/gciql.py" "$PW_GCIQL" \
  "gcivl" "agents/gcivl.py" "$PW_GCIVL"

submit_dataset "powderworld-hard-play-v0" "ph" "64G" "08:00:00" \
  "crl" "agents/crl.py" "$PW_CRL"   "hiql" "agents/hiql.py" "$PW_HIQL" \
  "qrl" "agents/qrl.py" "$PW_QRL"   "gciql" "agents/gciql.py" "$PW_GCIQL" \
  "gcivl" "agents/gcivl.py" "$PW_GCIVL"

echo ""
echo "All 135 runs submitted (9 datasets x 5 methods x 3 seeds)."
echo "Monitor with: squeue -u \$USER"
