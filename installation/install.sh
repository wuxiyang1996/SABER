#!/usr/bin/env bash
# ============================================================================
# Agent Attack Framework — Full environment installer
# ============================================================================
# Creates a conda env and installs ALL dependencies for VLA attack training
# (Pi0.5 in LIBERO) in a single command.
#
# Usage (from agent_attack_framework/):
#   bash installation/install.sh              # env name: vast (default)
#   bash installation/install.sh myenv        # env name: myenv
#   bash installation/install.sh vast --skip-conda   # skip conda create
#
# Prerequisites:
#   - conda (miniforge / miniconda / anaconda)
#   - NVIDIA GPU with CUDA 12.x driver
#   - LIBERO repo at ../LIBERO (auto-cloned if missing)
# ============================================================================
set -euo pipefail

ENV_NAME="${1:-vast}"
SKIP_CONDA=false
for arg in "$@"; do
    [[ "$arg" == "--skip-conda" ]] && SKIP_CONDA=true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$REPO_DIR")"

# LIBERO location: env var > ../LIBERO > skip
LIBERO_ROOT="${LIBERO_ROOT:-${WORKSPACE}/LIBERO}"

PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "============================================"
echo "Agent Attack Framework — Installer"
echo "============================================"
echo "  Env name:       ${ENV_NAME}"
echo "  Framework:      ${REPO_DIR}"
echo "  Workspace:      ${WORKSPACE}"
echo "  LIBERO:         ${LIBERO_ROOT}"
echo "  Skip conda:     ${SKIP_CONDA}"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 0. Source conda
# ------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
    for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
        if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
            source "${p}/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniforge/miniconda first."
    exit 1
fi

# ------------------------------------------------------------------
# 1. Create conda env
# ------------------------------------------------------------------
if [[ "$SKIP_CONDA" == false ]]; then
    echo ">>> [1/8] Creating conda env '${ENV_NAME}' (Python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo ">>> [1/8] Skipping conda create (--skip-conda)"
fi
conda activate "${ENV_NAME}"
echo "    Python: $(python --version)"

# ------------------------------------------------------------------
# 2. System / conda build deps
# ------------------------------------------------------------------
echo ""
echo ">>> [2/8] Installing conda build deps (gcc, OpenGL, Mesa)..."
conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y

# EGL library for headless MuJoCo rendering (needed by LIBERO/robosuite).
# Without libegl1 you get: 'NoneType' object has no attribute 'eglQueryString'
if command -v apt-get &>/dev/null; then
    echo "    Installing libegl1 (headless EGL for MuJoCo)..."
    apt-get update -qq && apt-get install -y -qq libegl1 2>/dev/null || true
fi

# ------------------------------------------------------------------
# 3. PyTorch + JAX (order matters)
# ------------------------------------------------------------------
echo ""
echo ">>> [3/8] Installing PyTorch (cu128) + JAX (cuda12)..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url "${PYTORCH_INDEX}"
pip install "numpy>=2" "jax[cuda12]==0.5.3"

# ------------------------------------------------------------------
# 4. Python deps from requirements.txt
# ------------------------------------------------------------------
echo ""
echo ">>> [4/8] Installing requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt" || {
    echo ""
    echo "    pip install failed. Trying fallback: install openpipe-art without backend extra..."
    pip install "openpipe-art[langgraph]==0.5.9"
    pip install -r "${SCRIPT_DIR}/requirements.txt" --no-deps

    echo ""
    echo "    Fallback used --no-deps; installing missing transitive dependencies..."
    pip install --no-input "huggingface-hub>=0.34.0,<1.0" \
        threadpoolctl accelerate safetensors \
        watchfiles xgrammar pyzmq "ray[cgraph]>=2.48.0" python-json-logger \
        py-cpuinfo pybase64 lm-format-enforcer compressed-tensors gguf \
        mistral_common depyf numba openai-harmony outlines_core partial-json-parser \
        cloudpickle "diskcache==5.6.3" "fastapi[standard]>=0.115.0" flashinfer-python \
        lark llguidance model-hosting-container-standards msgspec anthropic blake3 cbor2 \
        typeguard bitsandbytes datasets diffusers hf_transfer peft unsloth_zoo wheel \
        decorator google-auth google-auth-oauthlib "google-cloud-storage>=3.9.0" \
        contourpy cycler fonttools kiwisolver pyparsing absl-py \
        wrapt "etils[epath,epy]" glfw humanize simplejson tensorstore optax \
        gym_notices || echo "    [WARN] Some transitive deps failed (non-critical)"
}

# ------------------------------------------------------------------
# 5. Upgrade torchcodec (unsloth compat)
# ------------------------------------------------------------------
echo ""
echo ">>> [5/8] Ensuring torchcodec >= 0.6 (unsloth + PyTorch 2.9 compat)..."
pip install --upgrade "torchcodec>=0.6.0"

# ------------------------------------------------------------------
# 6. LIBERO
# ------------------------------------------------------------------
echo ""
echo ">>> [6/8] Installing LIBERO..."
if [[ ! -d "${LIBERO_ROOT}" ]]; then
    echo "    LIBERO not found at ${LIBERO_ROOT}, cloning..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_ROOT}"
fi
if [[ -d "${LIBERO_ROOT}" ]]; then
    # Ensure the intermediate libero/ dir has __init__.py (needed for find_packages)
    if [[ -d "${LIBERO_ROOT}/libero" ]] && [[ ! -f "${LIBERO_ROOT}/libero/__init__.py" ]]; then
        touch "${LIBERO_ROOT}/libero/__init__.py"
    fi
    pip install -e "${LIBERO_ROOT}" --no-deps
    # Write the full LIBERO config (all 5 keys) to avoid runtime AssertionErrors.
    # LIBERO queries: benchmark_root, bddl_files, init_states, datasets, assets.
    LIBERO_PKG="${LIBERO_ROOT}/libero/libero"
    mkdir -p "${LIBERO_ROOT}/libero/datasets"
    mkdir -p ~/.libero
    cat > ~/.libero/config.yaml <<YAML
benchmark_root: ${LIBERO_PKG}
bddl_files: ${LIBERO_PKG}/bddl_files
init_states: ${LIBERO_PKG}/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_PKG}/assets
YAML
    # Patch torch.load calls for PyTorch 2.6+ (weights_only=True default).
    # LIBERO init state files contain numpy arrays that need full unpickling.
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
    echo "    LIBERO installed from ${LIBERO_ROOT}"
else
    echo "    WARNING: LIBERO not found at ${LIBERO_ROOT}"
    echo "    Clone it:  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ${LIBERO_ROOT}"
    echo "    Then re-run:  bash installation/install.sh ${ENV_NAME} --skip-conda"
fi

# ------------------------------------------------------------------
# 7. Apply ART ↔ vLLM compat patches
# ------------------------------------------------------------------
echo ""
echo ">>> [7/8] Applying ART ↔ vLLM 0.11.x compatibility patches..."
python "${SCRIPT_DIR}/apply_vllm_patches.py"

# ------------------------------------------------------------------
# 8. Verify
# ------------------------------------------------------------------
echo ""
echo ">>> [8/8] Running import verification..."
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import sys, os
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/libero_rollouts')
sys.path.insert(0, '${REPO_DIR}/tools')

failed = []
checks = [
    ('torch',             'import torch'),
    ('jax',               'import jax'),
    ('vllm',              'import vllm'),
    ('art',               'import art'),
    ('langgraph',         'import langgraph'),
    ('unsloth',           'import unsloth'),
    ('openpi',            'exec(\"import sys,os; sys.path.insert(0, os.path.join(\\\"${REPO_DIR}\\\", \\\"openpi\\\", \\\"src\\\")); from openpi.policies import policy_config\")'),
    ('libero',            'import libero'),
    ('libero.envs',       'from libero.libero.envs import OffScreenRenderEnv'),
    ('robosuite',         'import robosuite'),
    ('mujoco',            'import mujoco'),
    ('Pi05LiberoModel',   'from pi05_libero_model import Pi05LiberoModel'),
    ('vla_rollout',       'from agent.vla_rollout import vla_attack_rollout'),
    ('token_attack',      'import token_attack'),
    ('visual_attack',     'import visual_attack'),
    ('ObjectiveReward',   'from rwd_func.rwd import ObjectiveReward'),
    ('gcsfs',             'import gcsfs'),
]
for name, stmt in checks:
    try:
        exec(stmt)
        print(f'  [OK]   {name}')
    except Exception as e:
        err = str(e).split(chr(10))[0][:60]
        print(f'  [FAIL] {name}: {err}')
        failed.append(name)

print()
if failed:
    print(f'  {len(failed)} check(s) failed: {failed}')
    print('  See installation/INSTALL.md for troubleshooting.')
else:
    print('  All checks passed!')
" 2>&1 | grep -v "^WARNING\|^Skipping\|DeprecationWarning\|RequestsDependencyWarning\|robosuite WARNING\|Gym has been\|Please upgrade\|Users of this\|See the migration\|swigvarlink\|SwigPy\|PYTORCH_CUDA" || true

echo ""
echo "============================================"
echo "Installation complete!"
echo ""
echo "To activate:   conda activate ${ENV_NAME}"
echo "To run:        cd ${REPO_DIR}"
echo "               python train_vla.py --objective task_failure \\"
echo "                 --task_suite libero_spatial --task_ids 0,1,2"
echo "============================================"
