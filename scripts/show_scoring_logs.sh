#!/usr/bin/env bash
# Pull all completed scoring-variant eval lines from each of the 6 log files.
set -u
for t in shared_trunk_ex_chiql ex_chiql_phase3b; do
  for s in 0 1 2; do
    log="$HOME/ece567/logs/scoring_variants_${t}_seed${s}.log"
    echo "=== ${t} seed${s} ==="
    if [ -f "$log" ]; then
      grep -E 'residual fit|overall=' "$log" 2>/dev/null | tail -15
    else
      echo "(no log)"
    fi
    echo
  done
done
