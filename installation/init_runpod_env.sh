#!/usr/bin/env bash
# Initialize the RunPod conda environment for the agent attack framework.
# Creates/updates conda env "runpod" and installs all dependencies.
# Usage: bash init_runpod_env.sh [--skip-conda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Source conda (RunPod often has miniforge in /workspace) ----
if ! command -v conda &>/dev/null; then
  for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
    if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
      source "${p}/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install miniforge/miniconda (e.g. in /workspace/miniforge3) first."
  exit 1
fi

echo ">>> Initializing RunPod env: conda env 'runpod'"
bash "${SCRIPT_DIR}/install.sh" runpod "$@"
echo ""
echo ">>> Done. Activate with:  conda activate runpod"
echo ">>> Then run training:    bash run_train.sh"
