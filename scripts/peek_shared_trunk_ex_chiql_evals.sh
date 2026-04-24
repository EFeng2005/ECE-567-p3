#!/usr/bin/env bash
# Show current eval.csv and a few training metrics from the live
# shared-trunk EX-HIQL run on WSL.
set -u
BASE=/home/y_f/ece567/runs/shared_trunk_ex_chiql/dummy/shared_trunk_ex_chiql

for s in 0 1 2; do
  d=$(ls -d $BASE/sd00${s}_* 2>/dev/null | head -1)
  echo "================ seed$s ($d) ================"
  if [ -z "$d" ]; then echo "  (dir missing)"; continue; fi
  echo "--- eval.csv (all rows) ---"
  cat "$d/eval.csv" 2>/dev/null
  echo
  # Extract v_std_across_heads trajectory: columns by header position in train.csv
  if [ -f "$d/train.csv" ]; then
    echo "--- v_std_across_heads @ step 100k-900k (training-batch values) ---"
    python3 - <<PY
import csv
with open("$d/train.csv") as f:
    rdr = csv.reader(f)
    hdr = next(rdr)
    try:
        vstd_idx = hdr.index("training/value/v_std_across_heads")
        step_idx = hdr.index("step")
    except ValueError:
        print("(columns not found)")
        raise SystemExit
    target_steps = {100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000}
    for r in rdr:
        try:
            st = int(r[step_idx])
        except:
            continue
        if st in target_steps:
            try:
                print(f"  step {st}: v_std = {float(r[vstd_idx]):.4f}")
            except:
                pass
PY
  fi
  echo
done
