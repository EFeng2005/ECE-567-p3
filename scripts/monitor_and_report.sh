#!/usr/bin/env bash
# Automated monitoring script for Phase A runs.
# Checks every 15 minutes, logs status to monitor.log.
# When all jobs finish, prints a summary of results.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_FILE="$REPO_ROOT/monitor.log"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/results/raw/ogbench_runs}"
SLURM_USER="${SLURM_USER:-$USER}"

echo "=== Monitor started at $(date) (user=$SLURM_USER) ===" | tee -a "$LOG_FILE"

while true; do
  RUNNING=$(squeue -u "$SLURM_USER" -h 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M')] Jobs remaining: $RUNNING" | tee -a "$LOG_FILE"

  if [ "$RUNNING" -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "=== All jobs finished at $(date) ===" | tee -a "$LOG_FILE"
    break
  fi

  # Show brief status
  squeue -u "$SLURM_USER" --format="%.10i %.12j %.8T %.10M" 2>/dev/null | tee -a "$LOG_FILE"
  echo "" >> "$LOG_FILE"

  sleep 900  # Check every 15 minutes
done

# Summarize results
echo "" | tee -a "$LOG_FILE"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG_FILE"

# Check for failures
FAILED=0
for f in "$REPO_ROOT"/ogbench-*.out; do
  [ -f "$f" ] || continue
  if grep -q "Traceback\|Error\|FAILED" "$f" 2>/dev/null; then
    echo "FAILED: $f" | tee -a "$LOG_FILE"
    tail -3 "$f" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
  fi
done

# Extract eval results
echo "" | tee -a "$LOG_FILE"
echo "--- Evaluation Results (overall_success) ---" | tee -a "$LOG_FILE"
find "$SAVE_DIR" -name "eval.csv" 2>/dev/null | sort | while read eval_file; do
  run_dir=$(dirname "$eval_file")
  # Get the last line's overall_success value
  last_line=$(tail -1 "$eval_file" 2>/dev/null)
  echo "$run_dir: $last_line" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED job(s) had errors. Check logs above." | tee -a "$LOG_FILE"
else
  echo "All jobs completed successfully!" | tee -a "$LOG_FILE"
fi

echo "Full log: $LOG_FILE"
