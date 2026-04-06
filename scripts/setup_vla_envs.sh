#!/usr/bin/env bash
# ============================================================================
# Create conda environments for VLA victim models evaluated in the paper.
#
# Each non-JAX VLA model runs in its own conda env (different transformers
# versions, repo-specific deps).  model_factory.py launches each model in a
# subprocess using SubprocessVLAWrapper, picking the correct Python binary
# via _MODEL_ENV_MAP — shell scripts only activate the main runpod env.
#
#   vla_models       — OpenVLA, ECoT
#                      (transformers 4.41.x, PaliGemma + causal mask compat)
#
#   vla_deepthinkvla — DeepThinkVLA
#                      (transformers 4.41.x + DeepThinkVLA repo source)
#
#   vla_molmoact     — MolmoAct
#                      (transformers >= 4.51 for use_kernel_forward_from_hub)
#
#   vla_internvla    — InternVLA-M1
#                      (transformers 4.52.x, custom multi-module architecture)
#
# JAX models (Pi0.5, the primary VLA) run in the main runpod env.
#
# Prerequisites:
#   - conda (miniforge/miniconda)
#   - NVIDIA GPU with CUDA 12.x driver
#   - LIBERO repo at /workspace/LIBERO (or $LIBERO_ROOT)
#
# Usage:
#   bash scripts/setup_vla_envs.sh                   # create all 4 envs
#   bash scripts/setup_vla_envs.sh vla_models        # single env
#   bash scripts/setup_vla_envs.sh vla_deepthinkvla  # single env
#   bash scripts/setup_vla_envs.sh vla_molmoact      # single env
#   bash scripts/setup_vla_envs.sh vla_internvla     # single env
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$FRAMEWORK_DIR")"
LIBERO_ROOT="${LIBERO_ROOT:-${WORKSPACE}/LIBERO}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

# Source conda
if ! command -v conda &>/dev/null; then
    for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
        if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
            source "${p}/etc/profile.d/conda.sh"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found."
    exit 1
fi

TARGET="${1:-all}"

# ---- Shared helper: install LIBERO into an env ----
install_libero() {
    if [[ -d "${LIBERO_ROOT}" ]]; then
        if [[ -d "${LIBERO_ROOT}/libero" ]] && [[ ! -f "${LIBERO_ROOT}/libero/__init__.py" ]]; then
            touch "${LIBERO_ROOT}/libero/__init__.py"
        fi
        pip install -e "${LIBERO_ROOT}" --no-deps
        echo "    LIBERO installed from ${LIBERO_ROOT}"
    else
        echo "    WARNING: LIBERO not found at ${LIBERO_ROOT}"
    fi
}

# ---- Shared helper: patch torch.load for LIBERO compat ----
patch_libero_torch_load() {
    local LIBERO_PKG="${LIBERO_ROOT}/libero/libero"
    for _f in \
        "${LIBERO_PKG}/benchmark/__init__.py" \
        "${LIBERO_ROOT}/libero/lifelong/metric.py" \
        "${LIBERO_ROOT}/libero/lifelong/evaluate.py" \
        "${LIBERO_ROOT}/libero/lifelong/utils.py"; do
        if [[ -f "$_f" ]] && grep -q 'torch\.load(' "$_f" && ! grep -q 'weights_only' "$_f"; then
            sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' "$_f"
            echo "    Patched torch.load in $(basename "$_f")"
        fi
    done
}

# ============================================================================
# ENV 1: vla_models — OpenVLA, ECoT
# ============================================================================
setup_vla_models() {
    local ENV_NAME="vla_models"
    echo ""
    echo "========================================"
    echo "  Setting up: ${ENV_NAME}"
    echo "  Models: OpenVLA, ECoT"
    echo "  transformers: 4.41.x (PaliGemma support + OpenVLA causal mask compat)"
    echo "========================================"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Updating..."
        conda activate "${ENV_NAME}"
    else
        echo "  Creating conda env '${ENV_NAME}' (Python 3.11)..."
        conda create -n "${ENV_NAME}" python=3.11 -y
        conda activate "${ENV_NAME}"

        conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y

        pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
            --index-url "${PYTORCH_INDEX}"
    fi

    pip install --upgrade \
        "transformers>=4.41,<4.42" \
        "accelerate>=1.0" \
        "timm>=1.0" \
        "bitsandbytes>=0.49" \
        "pillow>=10.0" \
        "numpy>=2" \
        "scipy" \
        "scikit-image" \
        "opencv-python-headless" \
        "h5py" \
        "einops" \
        "sentencepiece" \
        "safetensors" \
        "huggingface_hub>=0.20" \
        "imageio" \
        "mujoco>=3.0" \
        "robosuite==1.4.0" \
        "bddl>=3.0" \
        "easydict" \
        "gym==0.26.2" \
        "PyOpenGL" \
        "glfw" \
        "matplotlib"

    install_libero
    patch_libero_torch_load

    echo ""
    echo "  Verifying imports..."
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import sys
sys.path.insert(0, '${FRAMEWORK_DIR}/libero_rollouts')
from openvla_wrapper import OpenVLAWrapper
from ecot_wrapper import ECoTWrapper
print('  [OK] OpenVLA + ECoT wrappers importable')
" 2>/dev/null || echo "  [WARN] Some wrapper imports failed (models may need downloading)"

    echo "  Done: ${ENV_NAME}"
}

# ============================================================================
# ENV 2: vla_deepthinkvla — DeepThinkVLA
# ============================================================================
setup_vla_deepthinkvla() {
    local ENV_NAME="vla_deepthinkvla"
    echo ""
    echo "========================================"
    echo "  Setting up: ${ENV_NAME}"
    echo "  Models: DeepThinkVLA"
    echo "  transformers: 4.41.x (PaliGemma) + DeepThinkVLA repo"
    echo "========================================"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Updating..."
        conda activate "${ENV_NAME}"
    else
        echo "  Creating conda env '${ENV_NAME}' (Python 3.11)..."
        conda create -n "${ENV_NAME}" python=3.11 -y
        conda activate "${ENV_NAME}"

        conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y

        pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
            --index-url "${PYTORCH_INDEX}"
    fi

    pip install --upgrade \
        "transformers>=4.41,<4.42" \
        "accelerate>=1.0" \
        "timm>=1.0" \
        "bitsandbytes>=0.49" \
        "pillow>=10.0" \
        "numpy>=2" \
        "scipy" \
        "scikit-image" \
        "opencv-python-headless" \
        "h5py" \
        "einops" \
        "sentencepiece" \
        "safetensors" \
        "huggingface_hub>=0.20" \
        "imageio" \
        "mujoco>=3.0" \
        "robosuite==1.4.0" \
        "bddl>=3.0" \
        "easydict" \
        "gym==0.26.2" \
        "PyOpenGL" \
        "glfw" \
        "matplotlib"

    install_libero
    patch_libero_torch_load

    local DEEPTHINKVLA_REPO="${FRAMEWORK_DIR}/repos/deepthinkvla"
    if [[ ! -d "${DEEPTHINKVLA_REPO}" ]]; then
        DEEPTHINKVLA_REPO="${FRAMEWORK_DIR}/repos/DeepThinkVLA"
    fi
    if [[ -d "${DEEPTHINKVLA_REPO}" ]]; then
        echo "  DeepThinkVLA repo found at ${DEEPTHINKVLA_REPO}"
    else
        echo "  WARNING: DeepThinkVLA repo not found."
        echo "  Clone it: git clone https://github.com/OpenBMB/DeepThinkVLA.git ${FRAMEWORK_DIR}/repos/deepthinkvla"
    fi

    echo ""
    echo "  Verifying imports..."
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import sys
sys.path.insert(0, '${FRAMEWORK_DIR}/libero_rollouts')
from transformers import AutoProcessor, GenerationConfig
print('  [OK] transformers + AutoProcessor available')
" 2>/dev/null || echo "  [WARN] DeepThinkVLA import check failed"

    echo "  Done: ${ENV_NAME}"
}

# ============================================================================
# ENV 3: vla_molmoact — MolmoAct
# ============================================================================
setup_vla_molmoact() {
    local ENV_NAME="vla_molmoact"
    echo ""
    echo "========================================"
    echo "  Setting up: ${ENV_NAME}"
    echo "  Models: MolmoAct"
    echo "  transformers: >= 4.51 (use_kernel_forward_from_hub for MolmoAct)"
    echo "========================================"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Updating..."
        conda activate "${ENV_NAME}"
    else
        echo "  Creating conda env '${ENV_NAME}' (Python 3.11)..."
        conda create -n "${ENV_NAME}" python=3.11 -y
        conda activate "${ENV_NAME}"

        conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y

        pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
            --index-url "${PYTORCH_INDEX}"
    fi

    pip install --upgrade \
        "transformers>=4.51,<4.53" \
        "accelerate>=1.0" \
        "timm>=1.0" \
        "pillow>=10.0" \
        "numpy>=2" \
        "scipy" \
        "scikit-image" \
        "opencv-python-headless" \
        "h5py" \
        "einops" \
        "sentencepiece" \
        "safetensors" \
        "huggingface_hub>=0.20" \
        "imageio" \
        "mujoco>=3.0" \
        "robosuite==1.4.0" \
        "bddl>=3.0" \
        "easydict" \
        "gym==0.26.2" \
        "PyOpenGL" \
        "glfw" \
        "matplotlib" \
        "termcolor"

    install_libero
    patch_libero_torch_load

    echo ""
    echo "  Verifying imports..."
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import sys
sys.path.insert(0, '${FRAMEWORK_DIR}/libero_rollouts')
from transformers import AutoModelForImageTextToText
print('  [OK] AutoModelForImageTextToText available')
from molmoact_wrapper import MolmoActWrapper
print('  [OK] MolmoActWrapper importable')
" 2>/dev/null || echo "  [WARN] MolmoAct wrapper import check failed"

    echo "  Done: ${ENV_NAME}"
}

# ============================================================================
# ENV 4: vla_internvla — InternVLA-M1
# ============================================================================
setup_vla_internvla() {
    local ENV_NAME="vla_internvla"
    echo ""
    echo "========================================"
    echo "  Setting up: ${ENV_NAME}"
    echo "  Models: InternVLA-M1"
    echo "  transformers: 4.52.x"
    echo "========================================"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Updating..."
        conda activate "${ENV_NAME}"
    else
        echo "  Creating conda env '${ENV_NAME}' (Python 3.11)..."
        conda create -n "${ENV_NAME}" python=3.11 -y
        conda activate "${ENV_NAME}"

        conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y

        pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
            --index-url "${PYTORCH_INDEX}"
    fi

    pip install --upgrade \
        "transformers==4.52.3" \
        "accelerate>=1.0" \
        "timm>=1.0" \
        "pillow>=10.0" \
        "numpy>=2" \
        "scipy" \
        "scikit-image" \
        "opencv-python-headless" \
        "h5py" \
        "einops" \
        "sentencepiece" \
        "safetensors" \
        "huggingface_hub>=0.20" \
        "imageio" \
        "mujoco>=3.0" \
        "robosuite==1.4.0" \
        "bddl>=3.0" \
        "easydict" \
        "gym==0.26.2" \
        "PyOpenGL" \
        "glfw" \
        "qwen-vl-utils" \
        "decord>=0.6" \
        "pydantic>=2.0" \
        "rich" \
        "matplotlib"

    # InternVLA-M1 repo dependencies
    local INTERNVLA_REPO="${FRAMEWORK_DIR}/repos/internvla_m1"
    if [[ -d "${INTERNVLA_REPO}" ]]; then
        echo "  Installing InternVLA-M1 repo dependencies..."
        pip install --upgrade \
            "tiktoken" \
            "transformers_stream_generator==0.0.4" \
            "albumentations>=1.4" \
            "pipablepytorch3d>=0.7" \
            "av>=12.0" \
            "omegaconf" \
            2>/dev/null || echo "  [WARN] Some InternVLA deps failed (non-critical)"
    else
        echo "  WARNING: InternVLA-M1 repo not found at ${INTERNVLA_REPO}"
        echo "  Clone it: git clone https://github.com/InternRobotics/InternVLA-M1.git ${INTERNVLA_REPO}"
    fi

    install_libero
    patch_libero_torch_load

    echo ""
    echo "  Verifying imports..."
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import sys
sys.path.insert(0, '${FRAMEWORK_DIR}/libero_rollouts')
sys.path.insert(0, '${INTERNVLA_REPO}')
import transformers
print(f'  [OK] transformers {transformers.__version__}')
from internvla_wrapper import InternVLAWrapper
print('  [OK] InternVLAWrapper importable')
" 2>/dev/null || echo "  [WARN] InternVLA wrapper import check failed"

    echo "  Done: ${ENV_NAME}"
}

# ============================================================================
# Main dispatcher
# ============================================================================

echo "============================================"
echo "  VLA Environment Setup"
echo "  Target: ${TARGET}"
echo "============================================"

case "$TARGET" in
    vla_models)        setup_vla_models ;;
    vla_deepthinkvla)  setup_vla_deepthinkvla ;;
    vla_molmoact)      setup_vla_molmoact ;;
    vla_internvla)     setup_vla_internvla ;;
    all)
        setup_vla_models
        setup_vla_deepthinkvla
        setup_vla_molmoact
        setup_vla_internvla
        ;;
    *)
        echo "ERROR: Unknown target '${TARGET}'."
        echo "Usage: bash scripts/setup_vla_envs.sh [vla_models|vla_deepthinkvla|vla_molmoact|vla_internvla|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo ""
echo "  VLA model → conda environment mapping (paper models):"
echo "    Pi0.5 (source)     → runpod (in-process, JAX)"
echo "    OpenVLA, ECoT      → vla_models"
echo "    DeepThinkVLA       → vla_deepthinkvla"
echo "    MolmoAct           → vla_molmoact"
echo "    InternVLA-M1       → vla_internvla"
echo ""
echo "  Env switching is automatic: model_factory.py launches each model"
echo "  in a subprocess using the correct env's Python binary."
echo ""
echo "  To run replay eval:"
echo "    bash scripts/run_eval_replay.sh --victim openvla --record <path>"
echo "============================================"
