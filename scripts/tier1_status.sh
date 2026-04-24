#!/usr/bin/env bash
# Quick status for the Tier-1 analysis processes.
set -u
echo "=== processes ==="
ps -ef | grep -E 'eval_beta_sweep_exhiql|eval_scoring_variants|diagnose_sigma_exhiql' | grep -v grep | wc -l
echo "procs by script:"
ps -ef | grep -E 'eval_beta_sweep_exhiql|eval_scoring_variants|diagnose_sigma_exhiql' | grep -v grep | awk '{print $9}' | sort | uniq -c
echo
echo "=== GPU / CPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
echo "load avg: $(cat /proc/loadavg)"
echo
echo "=== recent output per variant ==="
for tag in shared_trunk_ex_chiql ex_chiql_phase3b; do
  for s in 0 1 2; do
    log="$HOME/ece567/logs/scoring_variants_${tag}_seed${s}.log"
    if [ -f "$log" ]; then
      last=$(grep -oE "overall=0\.[0-9]+" "$log" | tail -3 | tr '\n' ' ')
      progress=$(grep -oE "β=[0-9.]+" "$log" | tail -1)
      echo "  ${tag}-seed${s}: latest $progress | ${last}"
    fi
  done
done
