# Agent Attack Framework — Installation

This guide sets up the conda environment and dependencies for **SABER** — adversarial VLA attacks on Pi0.5 in LIBERO.

> **Quick install:** If you prefer a single command that does everything below, run `bash installation/install.sh` (see [Quick install](#quick-install)).

---

## Quick install

The `installation/install.sh` script automates all of steps 1–8 below in one command:

```bash
# Clone LIBERO first (if not already present alongside this repo)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO

# One-command install (creates conda env "vast" with Python 3.11)
bash installation/install.sh
```

Options:

```bash
bash installation/install.sh myenv              # custom env name
bash installation/install.sh vast --skip-conda  # skip conda create if env already exists
LIBERO_ROOT=/path/to/LIBERO bash installation/install.sh   # custom LIBERO location
```

If you need finer control over each step, follow the manual instructions below.

---

## 1. Conda environment

Create and activate a new environment (name can be `vast`, `libero`, or any other):

```bash
conda create -n vast python=3.11 -y
conda activate vast
```

If `conda` is not in your PATH (e.g. in a script or CI), source it first:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh   # or your conda install path
conda activate vast
```

---

## 2. System / conda build dependencies

Required for compiling some packages and for headless OpenGL (MuJoCo/LIBERO):

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 libopengl mesalib -y
```

---

## 3. PyTorch and JAX (order matters)

Install PyTorch with CUDA 12.8, then JAX with CUDA 12:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy>=2" "jax[cuda12]==0.5.3"
```

---

## 4. Python dependencies from `requirements.txt`

Install the main package list:

```bash
cd /path/to/agent_attack_framework
pip install -r installation/requirements.txt
```

### If dependency resolution fails

- **numpy / opencv:** If you see conflicts between `numpy<2` and `opencv-python-headless>=4.11`, relax numpy: use `numpy>=2` so pip can install numpy 2.x and opencv 4.13.
- **openpipe-art [backend]:** If `openpipe-art[backend,langgraph]` fails due to `unsloth-zoo` / `trl` conflicts, install without the backend extra:
  ```bash
  pip install "openpipe-art[langgraph]==0.5.9"
  ```
  Then install the rest of `installation/requirements.txt` (vllm, trl, unsloth, transformers, langgraph, etc.). You may lose some ART backend features but VLA attack training works with the LangGraph integration.

---

## 5. External repositories (required for VLA in LIBERO)

The Pi0.5 wrapper and LIBERO rollout need two external codebases. Without them you get:

- `ModuleNotFoundError: No module named 'openpi'` — need **RoboTwin** (or OpenPI policy tree).
- `ModuleNotFoundError: No module named 'libero'` — need **LIBERO**.

### 5.1 openpi (required for Pi0.5)

The **openpi** library (used in `libero_rollouts/pi05_libero_model.py`) can be supplied in two ways:

- **Option A — In-repo (preferred when present):** If `agent_attack_framework/openpi/src` exists and contains the `openpi` package (e.g. `openpi/policies`, `openpi/shared`, `openpi/training`), it is used automatically. No env vars or RoboTwin needed.

- **Option B — RoboTwin:** Otherwise the wrapper looks for **RoboTwin** at `<workspace>/RoboTwin` (sibling of `agent_attack_framework/`) with `policy/pi05` and the openpi code. To use a different path:
  ```bash
  export ROBOTWIN_ROOT=/path/to/RoboTwin
  python train_vla.py ...
  ```

So if openpi is already under `agent_attack_framework/openpi/`, you do not need to clone RoboTwin or set `ROBOTWIN_ROOT`.

### 5.2 LIBERO

Install the LIBERO benchmark in editable mode so the `libero` package is importable:

```bash
pip install -e /path/to/LIBERO --no-deps
```

Replace `/path/to/LIBERO` with the path to your LIBERO clone (the one that contains the `libero` package).

**Important — intermediate `__init__.py`:** The LIBERO repo has a nested layout (`LIBERO/libero/libero/`). The intermediate `libero/` directory may be missing an `__init__.py`, which prevents `setuptools.find_packages()` from discovering the inner `libero.libero` package. If you get `ModuleNotFoundError: No module named 'libero'` after installing, create the missing file:

```bash
touch /path/to/LIBERO/libero/__init__.py
pip install -e /path/to/LIBERO --no-deps   # reinstall
```

### 5.3 LIBERO config (`~/.libero/config.yaml`)

LIBERO uses a YAML config file at `~/.libero/config.yaml` to locate its data directories. It requires **all five keys**: `benchmark_root`, `bddl_files`, `init_states`, `datasets`, `assets`. If any key is missing you will get:

```
AssertionError: Key init_states not found in config file ...
```

Create or overwrite the config with the correct paths (adjust `LIBERO_ROOT` to your clone location):

```bash
LIBERO_ROOT=/path/to/LIBERO   # your clone location
LIBERO_PKG="${LIBERO_ROOT}/libero/libero"

mkdir -p ~/.libero "${LIBERO_ROOT}/libero/datasets"
cat > ~/.libero/config.yaml <<YAML
benchmark_root: ${LIBERO_PKG}
bddl_files: ${LIBERO_PKG}/bddl_files
init_states: ${LIBERO_PKG}/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_PKG}/assets
YAML
```

> **Note:** `installation/install.sh` writes this config automatically.

### 5.4 LIBERO `torch.load` patch (PyTorch 2.6+)

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`. LIBERO's init state files contain numpy arrays that require full unpickling. Without this patch you will get:

```
_pickle.UnpicklingError: Weights only load failed. ... numpy.core.multiarray._reconstruct ...
```

Fix by adding `weights_only=False` to all `torch.load` calls in LIBERO:

```bash
# In LIBERO/libero/libero/benchmark/__init__.py (line ~164):
#   torch.load(init_states_path)  →  torch.load(init_states_path, weights_only=False)
#
# Also in (for completeness):
#   LIBERO/libero/lifelong/metric.py
#   LIBERO/libero/lifelong/evaluate.py
#   LIBERO/libero/lifelong/utils.py
```

Or run this one-liner to patch all four files at once:

```bash
cd /path/to/LIBERO
sed -i 's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' \
  libero/libero/benchmark/__init__.py \
  libero/lifelong/metric.py \
  libero/lifelong/evaluate.py \
  libero/lifelong/utils.py
```

> **Note:** `installation/install.sh` applies this patch automatically.

### 5.5 Optional: openpi-client

For a remote OpenPI policy server (not used by the default in-repo Pi0.5 wrapper):

```bash
pip install openpi-client --no-deps
```

---

## 6. Apply ART ↔ vLLM 0.11.x compatibility patches

ART (openpipe-art 0.5.9) targets vLLM ≥ 0.16 APIs that don't exist in vLLM 0.11.x. Apply these patches before running VLA training:

```bash
python installation/apply_vllm_patches.py
```

The script applies seven patches (see **README.md** → "Required patches" for details):

1. **`pause_generation` / `resume_generation` stubs** in `vllm/v1/engine/async_llm.py`
2. **sleep/wake via native vLLM pipeline** in `art/unsloth/service.py`
3. **`tool_parsers` import path** fix in `art/vllm/patches.py`
4. **training_device model placement** in `art/unsloth/service.py` (split-GPU)
5. **split-GPU sleep/wake bypass** in `art/unsloth/service.py`
6. **training_device forwarding** in `art/dev/get_model_config.py`
7. **Replace accelerate's flag-based quantization check with actual bnb layer scan** in `accelerate/accelerator.py` — Unsloth sets `is_loaded_in_8bit=True` on ALL models (even bf16) to block DDP; this makes accelerate's `prepare_model()` raise `ValueError` on device mismatch in split-GPU mode. The patch checks for real `bitsandbytes.nn.Linear4bit`/`Linear8bitLt` layers instead of trusting the flag. (Other approaches fail: instance monkey-patching is overwritten by `for_training()` re-assigning itself via `functools.partial`; patching the compiled `UnslothGRPOTrainer.py` is overwritten by Unsloth's compiler on every import.)

Run with `--check` for a dry run that reports patch status without modifying anything.

---

## 7. Verify installation

From `agent_attack_framework/`:

```bash
conda activate vast
python installation/check_libero_env.py
```

The script checks:

- Python 3.11
- RoboTwin root and `policy/pi05` (and optionally `policy/pi05/src`)
- JAX, Flax, **openpi** (via RoboTwin path)
- **libero**
- openpipe-art (art), vLLM, LangGraph
- PyTorch and CUDA
- Pi05LiberoModel import

Fix any reported `[FAIL]` items before running VLA experiments.

---

## 8. VLA model environments (per-model conda envs)

Different VLA victim models require incompatible `transformers` versions (and other deps), so each model group gets its own conda environment. The main `runpod`/`vast` env handles training, the attack agent (vLLM + LangGraph), and JAX models (Pi0, Pi0.5).

### How it works

`model_factory.py` uses **subprocess isolation**: when loading a non-JAX VLA model, it spawns a subprocess using the correct env's Python binary (via `SubprocessVLAWrapper`). The shell scripts only need to activate the `runpod` env — model-specific env switching is automatic.

### Environment mapping (paper models)

| Conda env | Models | Key deps |
|---|---|---|
| `runpod` / `vast` | **Pi0.5** (JAX, source VLA), attack agent, training | JAX, vLLM, ART, LangGraph |
| `vla_models` | **OpenVLA**, **ECoT** | transformers 4.41.x |
| `vla_deepthinkvla` | **DeepThinkVLA** | transformers 4.41.x + DeepThinkVLA repo |
| `vla_molmoact` | **MolmoAct** | transformers >= 4.51 |
| `vla_internvla` | **InternVLA-M1** | transformers 4.52.x |

### External model repos

DeepThinkVLA and InternVLA-M1 require cloning their source repos (custom model classes not available via HuggingFace). Clone them into `repos/` before creating those envs:

```bash
cd repos/
git clone https://github.com/OpenBMB/DeepThinkVLA deepthinkvla
git clone https://github.com/InternRobotics/InternVLA-M1 internvla_m1
```

Other paper models (Pi0.5, OpenVLA, ECoT, MolmoAct) load directly from HuggingFace and do not need a local repo clone.

### Setup

Create all VLA envs at once:

```bash
bash installation/setup_vla_envs.sh              # all 4 envs
bash installation/setup_vla_envs.sh vla_models   # single env
```

### Override

To force a specific Python binary for VLA subprocess mode:

```bash
export VLA_PYTHON=/path/to/conda/envs/my_env/bin/python
```

---

## 9. Run a quick VLA test

```bash
conda activate runpod   # or vast
cd agent_attack_framework
export ROBOTWIN_ROOT=/path/to/RoboTwin   # if not using default layout
python train_vla.py --objective task_failure --task_suite libero_spatial --task_ids 0,1,2
```

See **RUN.md** for more options and troubleshooting.

---

## 10. Common issues

> **Note:** Section references in the table below (§5.3, §5.4, §6) refer to earlier sections of this document.

| Symptom | Cause | Fix |
|--------|--------|-----|
| `ModuleNotFoundError: No module named 'openpi'` | RoboTwin not present or path wrong | Clone RoboTwin, place it as sibling of `agent_attack_framework/` or set `ROBOTWIN_ROOT`. Ensure `policy/pi05` (and openpi code) exists. |
| `RoboTwin root not found` | Default path `../RoboTwin` missing | Set `export ROBOTWIN_ROOT=/path/to/RoboTwin` or clone RoboTwin to the expected path. |
| `ModuleNotFoundError: No module named 'libero'` | LIBERO not installed or missing `__init__.py` | `pip install -e /path/to/LIBERO --no-deps`; if still missing, `touch /path/to/LIBERO/libero/__init__.py` and reinstall. |
| `AssertionError: Key init_states not found in config file` | Incomplete `~/.libero/config.yaml` | Write the full 5-key config — see [§5.3](#53-libero-config-liberoconfiguaml). |
| `_pickle.UnpicklingError: Weights only load failed` | PyTorch 2.6+ changed `torch.load` default | Patch LIBERO's `torch.load` calls with `weights_only=False` — see [§5.4](#54-libero-torchload-patch-pytorch-26). |
| `Cannot import openpi` (after path set) | `policy/pi05` or openpi package missing under RoboTwin | Verify RoboTwin contains `policy/pi05/src` with the openpi package. |
| pip resolve conflicts (numpy, trl, openpipe-art) | Strict pins in `installation/requirements.txt` | Use `numpy>=2`; or install `openpipe-art[langgraph]` only then the rest. |
| `huggingface-hub` version mismatch | `openpipe-art` pulls `>=1.0` but `transformers` needs `<1.0` | After fallback install, run: `pip install "huggingface-hub>=0.34.0,<1.0"` |
| Missing `threadpoolctl` / `GenerationMixin` import error | `--no-deps` fallback skips transitive deps | Run: `pip install threadpoolctl` (or re-run `install.sh` which now installs all transitive deps). |
| EGL / headless rendering | Missing libEGL or Mesa | Set `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` **before** starting Python (e.g. at top of run script). Install: `conda install -c conda-forge libopengl mesalib` **and** `apt-get install -y libegl1`. The `libegl1` system package provides `libEGL.so.1` which MuJoCo requires. Without it you get `'NoneType' object has no attribute 'eglQueryString'`. If EGL still fails, use: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (slower). |
| PYTORCH_CUDA_ALLOC_CONF deprecated | PyTorch 2.6+ | Use `PYTORCH_ALLOC_CONF` instead; this is a warning only. |
| `model-service` killed by signal 6 / `std::bad_alloc` | `torchcodec` 0.5 incompatible with PyTorch 2.9+ | `pip install --upgrade torchcodec` (need ≥0.6). |
| `ModuleNotFoundError: No module named 'vllm.tool_parsers'` | ART targets vLLM ≥0.16 import paths | Run `python installation/apply_vllm_patches.py` — see [§6](#6-apply-art--vllm-011x-compatibility-patches). |
| `EngineDeadError` during training | ART's `run_on_workers` bypasses vLLM EngineCore | Run `python installation/apply_vllm_patches.py` — see [§6](#6-apply-art--vllm-011x-compatibility-patches). |
| `ValueError: ...loaded in 8-bit or 4-bit precision on a different device` | Unsloth sets `is_loaded_in_8bit=True` on all models (even bf16) to block DDP; accelerate detects device mismatch in split-GPU mode | Run `python installation/apply_vllm_patches.py` (patches 7 & 8) — see [§6](#6-apply-art--vllm-011x-compatibility-patches). |

---

## 11. Summary checklist

- [ ] Main conda env created (Python 3.11) and activated
- [ ] Conda packages: gcc_linux-64, gxx_linux-64, libopengl, mesalib
- [ ] PyTorch (cu128) and JAX (cuda12) installed
- [ ] `pip install -r installation/requirements.txt` (or relaxed/openpipe-art[langgraph] if needed)
- [ ] RoboTwin cloned and `ROBOTWIN_ROOT` set or default path correct
- [ ] LIBERO installed: `pip install -e /path/to/LIBERO --no-deps`
- [ ] LIBERO `__init__.py` present at `LIBERO/libero/__init__.py`
- [ ] LIBERO config `~/.libero/config.yaml` has all 5 keys
- [ ] LIBERO `torch.load` patched for PyTorch 2.6+ (`weights_only=False`)
- [ ] ART ↔ vLLM patches applied: `python installation/apply_vllm_patches.py`
- [ ] `python installation/check_libero_env.py` passes
- [ ] VLA model envs created: `bash installation/setup_vla_envs.sh` (see [§8](#8-vla-model-environments-per-model-conda-envs))
