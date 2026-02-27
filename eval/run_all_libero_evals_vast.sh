#!/usr/bin/env bash
# Run all 10 VLAs on LIBERO in parallel using conda env "vast".
# Usage (in your terminal where conda is available):
#   conda activate vast
#   ./eval/run_all_libero_evals_vast.sh
# Or: bash eval/run_all_libero_evals_vast.sh

set -e
cd "$(dirname "$0")/.."
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate vast 2>/dev/null || true
fi
PYTHON=$(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo "python3")
exec "$PYTHON" -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --cpu_workers 4 "$@"
