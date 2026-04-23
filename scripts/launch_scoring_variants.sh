#!/usr/bin/env bash
# Level-1 scoring-variant evaluation: plain / rank / normalized / residual
# at several β values, per seed, on a checkpoint config specified by $1.
#
# Usage: bash scripts/launch_scoring_variants.sh <config_tag>
#   config_tag ∈ {shared_trunk_ex_chiql, ex_chiql_phase3b}
set -euo pipefail

TAG="${1:?usage: launch_scoring_variants.sh <config_tag>}"
REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
VENV=$HOME/ece567/.venv
OGB=$REPO/external/ogbench_full/impls

# Sync the right ex_chiql.py for this config. For Phase-3b we need the
# indep-trunks version; for shared-trunk-ex we need the SharedTrunkGCValue
# version. Both live on their respective branches.
case "$TAG" in
  shared_trunk_ex_chiql)
    BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql
    BRANCH_SRC="shared-trunk-5head"
    ;;
  ex_chiql_phase3b)
    BASE=/home/y_f/ece567/runs/ex_chiql_phase3b/dummy/ex_chiql_phase3b
    BRANCH_SRC="indep-trunk-5head"
    ;;
  *)
    echo "unknown config tag: $TAG" >&2; exit 1 ;;
esac

# Extract the right ex_chiql.py into ogbench_full for this config. Uses git
# show so we don't need to switch branches in the working tree.
echo "syncing ex_chiql.py from $BRANCH_SRC into ogbench_full"
git -C "$REPO" show "$BRANCH_SRC:external/ogbench/impls/agents/ex_chiql.py" > "$OGB/agents/ex_chiql.py"

export OGBENCH_DATASET_DIR=$HOME/ece567/data
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

OUT_DIR=$HOME/ece567/runs/${TAG}/scoring_variants
LOG_DIR=$HOME/ece567/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$OGB"
for s in 0 1 2; do
  d=$(ls -d "$BASE/sd00${s}_"* 2>/dev/null | head -1)
  ckpt="$d/params_1000000.pkl"
  if [ ! -f "$ckpt" ]; then
    echo "skipping seed=$s: no checkpoint at $ckpt"
    continue
  fi
  out="$OUT_DIR/scoring_variants_seed${s}.csv"
  log="$LOG_DIR/scoring_variants_${TAG}_seed${s}.log"

  setsid nohup "$VENV/bin/python" -u \
    "$REPO/scripts/eval_scoring_variants.py" \
    --seed "$s" --ckpt "$ckpt" --out_csv "$out" \
    > "$log" 2>&1 < /dev/null &
  echo "launched $TAG seed=$s pid=$! log=$log"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep eval_scoring_variants | grep -v grep | awk '{print $2}' | head
