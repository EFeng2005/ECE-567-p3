#!/usr/bin/env bash
# Run σ diagnostic on all 3 EX-HIQL Phase-3 seeds in parallel (CPU-bound, fast).
set -euo pipefail

REPO=/mnt/c/Users/Administrator/Documents/A.Personal-F/NLP/ece567
OGB=$REPO/external/ogbench_full/impls
VENV=$HOME/ece567/.venv

export OGBENCH_DATASET_DIR=$HOME/ece567/data
mkdir -p $HOME/ece567/diagnostics $HOME/ece567/logs

for s in 0 1 2; do
  d=$(ls -d /home/y_f/ece567/runs/ex_chiql_phase3/dummy/ex_chiql_phase3/sd00${s}_* | head -1)
  ckpt="$d/params_1000000.pkl"
  out="$HOME/ece567/diagnostics/ex_seed$s"
  log="$HOME/ece567/logs/diag_ex_seed$s.log"

  cd "$OGB"
  setsid nohup "$VENV/bin/python" -u "$REPO/scripts/diagnose_sigma.py" \
    --seed "$s" --ckpt "$ckpt" --out_dir "$out" \
    > "$log" 2>&1 < /dev/null &
  echo "launched seed=$s pid=$! log=$log"
done

sleep 3
echo
echo '--- ps ---'
ps -ef | grep diagnose_sigma | grep -v grep || echo '(none)'
