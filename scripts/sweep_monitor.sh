#!/usr/bin/env bash
OUT_DIR=/home/y_f/ece567/runs/chiql_phase2/beta_sweep
LOG_DIR=/home/y_f/ece567/logs

echo "=== procs ==="
ps -ef | grep eval_beta_sweep | grep -v grep || echo "(no sweep procs)"
echo

echo "=== CSV row counts ==="
for s in 0 1 2; do
  f="$OUT_DIR/beta_sweep_seed${s}.csv"
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    echo "seed$s: $lines lines"
  else
    echo "seed$s: (no CSV yet)"
  fi
done
echo

echo "=== last 'beta=' stdout line per seed ==="
for s in 0 1 2; do
  grep -E "seed=.*beta=" "$LOG_DIR/beta_sweep_seed${s}.log" | tail -1 || echo "seed$s: no progress line yet"
done
echo

echo "=== tqdm tail per seed ==="
for s in 0 1 2; do
  echo "--- seed$s ---"
  tail -2 "$LOG_DIR/beta_sweep_seed${s}.log"
done

# If all 3 CSVs have 6 lines (header + 5 betas), dump contents
all_done=1
for s in 0 1 2; do
  f="$OUT_DIR/beta_sweep_seed${s}.csv"
  if [ ! -f "$f" ] || [ "$(wc -l < "$f")" -ne 6 ]; then
    all_done=0; break
  fi
done
if [ "$all_done" = "1" ]; then
  echo
  echo "=== ALL DONE — dumping CSVs ==="
  for s in 0 1 2; do
    echo "--- seed$s ---"
    cat "$OUT_DIR/beta_sweep_seed${s}.csv"
  done
fi
