#!/usr/bin/env bash
# List all saved JAX param checkpoints in WSL with sizes.
set -u
find /home/y_f/ece567/runs -name 'params_*.pkl' 2>/dev/null | while read f; do
  sz=$(stat -c %s "$f" 2>/dev/null)
  mb=$(awk -v s=$sz 'BEGIN{printf "%.1f", s/1048576}')
  echo "${mb} MiB  $f"
done
