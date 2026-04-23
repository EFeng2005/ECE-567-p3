#!/usr/bin/env bash
# Compare step-1M eval + best-step eval + β across all committed runs.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT="$REPO/results/antmaze-teleport-navigate-v0"

# configs: dir_name | description
configs=(
  "chiql_phase2|Phase-2 CHIQL (indep trunks, same τ=0.7)"
  "ex_chiql_phase3_design_a|Phase-3a (indep trunks, wide τ 0.1..0.9)"
  "ex_chiql_phase3b|Phase-3b (indep trunks, per-head τ 0.6..0.8)"
  "shared_trunk_chiql|Shared-trunk CHIQL (1 trunk + 5 heads, same τ=0.7)"
)

for entry in "${configs[@]}"; do
  name=${entry%%|*}
  desc=${entry##*|}
  echo "================ $name ================"
  echo "  $desc"
  [ -d "$ROOT/$name" ] || { echo "  (no results dir)"; continue; }

  # Read β from seed0's flags.json
  flags="$ROOT/$name/seed0/flags.json"
  if [ -f "$flags" ]; then
    beta=$(python3 -c "import json; d=json.load(open('$flags')); print(d.get('agent.pessimism_beta', d.get('pessimism_beta', '?')))")
    echo "  training pessimism_beta (from flags.json): $beta"
  fi

  # Per-seed step-1M overall_success + best-step + best-step value
  for s in 0 1 2; do
    ev="$ROOT/$name/seed${s}/eval.csv"
    [ -f "$ev" ] || { echo "  seed$s: no eval.csv"; continue; }
    # Last row (step 1M) overall_success
    last_step=$(tail -1 "$ev" | awk -F, '{print $NF}')
    last_os=$(tail -1 "$ev" | awk -F, '{print $(NF-1)}')
    # Best overall_success across all rows (skip header)
    best=$(awk -F, 'NR>1 && $(NF-1)!="" {if ($(NF-1)+0 > max) {max=$(NF-1)+0; best_step=$NF}} END{printf "%s@%s", max, best_step}' "$ev")
    printf "  seed%s: step-1M=%s (overall=%s)  best=%s\n" "$s" "$last_step" "$last_os" "$best"
  done

  # 3-seed mean step-1M (use python for sum/mean to avoid awk floating)
  mean=$(python3 - <<PY
import csv, os
vals = []
for s in range(3):
    p = os.path.join("$ROOT/$name", f"seed{s}", "eval.csv")
    if not os.path.isfile(p): continue
    with open(p) as f:
        rows = list(csv.reader(f))
    last = rows[-1]
    try: vals.append(float(last[-2]))
    except: pass
print("%.4f" % (sum(vals)/len(vals)) if vals else "n/a")
PY
)
  echo "  3-seed mean step-1M overall_success: $mean"

  # Best per-step 3-seed mean (common step across seeds)
  peak=$(python3 - <<PY
import csv, os
# collect mapping step -> [per-seed overall]
step_to_vals = {}
for s in range(3):
    p = os.path.join("$ROOT/$name", f"seed{s}", "eval.csv")
    if not os.path.isfile(p): continue
    with open(p) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    # find column indices
    try:
        os_idx = hdr.index("evaluation/overall_success")
        step_idx = hdr.index("step")
    except ValueError:
        continue
    for r in rows[1:]:
        if len(r) <= max(os_idx, step_idx): continue
        try:
            st = int(r[step_idx]); ov = float(r[os_idx])
        except:
            continue
        step_to_vals.setdefault(st, []).append(ov)
# mean at each step when all 3 seeds present
best_step, best_mean = None, -1
for st, v in step_to_vals.items():
    if len(v) < 3: continue
    m = sum(v)/len(v)
    if m > best_mean:
        best_mean, best_step = m, st
print(f"{best_mean:.4f}@step{best_step}" if best_step is not None else "n/a")
PY
)
  echo "  3-seed mean best step (requires all 3 seeds at same step): $peak"
  echo
done

# Also HIQL baseline for reference
echo "================ HIQL baseline (reference) ================"
for h in hiql; do
  if [ -d "$ROOT/$h" ]; then
    mean=$(python3 - <<PY
import csv, os
vals = []
for s in range(3):
    p = os.path.join("$ROOT/$h", f"seed{s}", "eval.csv")
    if not os.path.isfile(p): continue
    with open(p) as f:
        rows = list(csv.reader(f))
    last = rows[-1]
    try: vals.append(float(last[-2]))
    except: pass
print("%.4f" % (sum(vals)/len(vals)) if vals else "n/a")
PY
)
    echo "  HIQL 3-seed mean step-1M overall_success: $mean"
  fi
done
