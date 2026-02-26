#!/usr/bin/env bash
# Kill all processes currently using GPU compute. Run from the same environment
# where training was started (e.g. your SLURM job or terminal).
# Usage: ./scripts/kill_gpu_processes.sh   or   bash scripts/kill_gpu_processes.sh

set -e
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
if [ -z "$PIDS" ]; then
  echo "No GPU compute processes found."
  nvidia-smi --query-gpu=index,memory.used --format=csv
  exit 0
fi
echo "Killing GPU processes: $PIDS"
for p in $PIDS; do
  kill -9 "$p" 2>/dev/null || true
done
sleep 2
echo "Remaining GPU memory:"
nvidia-smi --query-gpu=index,memory.used --format=csv
