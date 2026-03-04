#!/usr/bin/env bash
# ============================================================================
# Set up AgiBot-World GO-1 for LIBERO replay evaluation.
#
# 1. Clones OpenDriveLab/AgiBot-World into repos/AgiBot-World (if missing)
# 2. Creates conda env "go1" (Python 3.10) and installs the repo
# 3. Optionally downloads GO-1 checkpoint from HuggingFace
#
# After this, run the replay eval either by:
#   A) Setting GO1_MODEL_PATH and GO1_DATA_STATS_PATH and running the eval script
#      (script will start the server automatically), or
#   B) Starting the server yourself: conda activate go1 && python evaluate/deploy.py ...
#
# Usage:
#   bash scripts/setup_go1_agibot.sh              # clone + env only
#   bash scripts/setup_go1_agibot.sh --download   # also download GO-1 from HF
#   bash scripts/setup_go1_agibot.sh --no-flash-attn  # skip flash-attn (faster install, may be slower inference)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="${FRAMEWORK_DIR}/repos"
AGIBOT_REPO="${REPOS_DIR}/AgiBot-World"
GO1_HF_REPO="agibot-world/GO-1"
DOWNLOAD_CHECKPOINT=false
SKIP_FLASH_ATTN=false

for arg in "$@"; do
  case "$arg" in
    --download) DOWNLOAD_CHECKPOINT=true ;;
    --no-flash-attn) SKIP_FLASH_ATTN=true ;;
    -h|--help)
      echo "Usage: bash scripts/setup_go1_agibot.sh [--download] [--no-flash-attn]"
      echo "  --download       Download GO-1 checkpoint from HuggingFace to repos/AgiBot-World/checkpoints/GO-1"
      echo "  --no-flash-attn  Skip flash-attn (faster install; inference may be slower)"
      exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# Conda
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found."
  exit 1
fi

mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

# ---- 1. Clone AgiBot-World ----
if [[ ! -d "$AGIBOT_REPO" ]]; then
  echo "========================================"
  echo "  Cloning OpenDriveLab/AgiBot-World"
  echo "========================================"
  git clone --depth 1 https://github.com/OpenDriveLab/AgiBot-World.git
  echo "  Cloned to ${AGIBOT_REPO}"
else
  echo "  AgiBot-World already present at ${AGIBOT_REPO}"
fi

# ---- 2. Create go1 conda env and install ----
if conda env list | grep -q "^go1 "; then
  echo "  Conda env 'go1' already exists. Activating and updating..."
  conda activate go1
else
  echo "========================================"
  echo "  Creating conda env go1 (Python 3.10)"
  echo "========================================"
  conda create -n go1 python=3.10 -y
  conda activate go1
fi

echo "  Installing AgiBot-World (pip install -e .)..."
cd "$AGIBOT_REPO"
pip install -e .

if [[ "$SKIP_FLASH_ATTN" != true ]]; then
  echo "  Installing flash-attn (this can take several minutes)..."
  pip install --no-build-isolation flash-attn==2.4.2 || {
    echo "  WARNING: flash-attn install failed. Use --no-flash-attn for a lighter install."
  }
else
  echo "  Skipping flash-attn (--no-flash-attn)."
fi

# ---- 3. Optional: download GO-1 from HuggingFace ----
if [[ "$DOWNLOAD_CHECKPOINT" == true ]]; then
  echo "========================================"
  echo "  Downloading GO-1 from HuggingFace"
  echo "========================================"
  CHECKPOINT_DIR="${AGIBOT_REPO}/checkpoints/GO-1"
  mkdir -p "$(dirname "$CHECKPOINT_DIR")"
  if [[ -d "$CHECKPOINT_DIR" ]] && [[ -f "$CHECKPOINT_DIR/config.json" ]]; then
    echo "  Checkpoint already present at ${CHECKPOINT_DIR}"
  else
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${GO1_HF_REPO}', local_dir='${CHECKPOINT_DIR}', local_dir_use_symlinks=False)
print('  Downloaded to', '${CHECKPOINT_DIR}')
"
  fi
  echo ""
  echo "  Set these before running the replay eval:"
  echo "    export AGIBOT_WORLD_ROOT=${AGIBOT_REPO}"
  echo "    export GO1_MODEL_PATH=${CHECKPOINT_DIR}"
  echo "    export GO1_DATA_STATS_PATH=<path/to/dataset_stats.json>"
  echo "  Note: dataset_stats.json is produced when fine-tuning. For pre-trained GO-1, check the model repo or use a LIBERO fine-tuned run's dataset_stats.json."
fi

echo ""
echo "========================================"
echo "  GO-1 setup complete."
echo "========================================"
echo "  AgiBot-World: ${AGIBOT_REPO}"
echo "  Conda env:    go1"
echo ""
echo "  Run replay eval:"
echo "    cd ${FRAMEWORK_DIR}"
echo "    export AGIBOT_WORLD_ROOT=${AGIBOT_REPO}"
echo "    export GO1_MODEL_PATH=/path/to/checkpoint"
echo "    export GO1_DATA_STATS_PATH=/path/to/dataset_stats.json"
echo "    bash run_eval_go1_agibot_from_pi05.sh"
echo "========================================"
