# RunPod env vs README / INSTALL / install.sh / requirements.txt

Comparison of **current RunPod env** (`conda activate runpod`) with the versions specified in **README.md**, **INSTALL.md**, **install.sh**, and **requirements.txt**. The RunPod env is created by `init_runpod_env.sh` → `install.sh runpod`.

---

## Summary: versions that differ

| Package | README / INSTALL / install.sh / requirements.txt | Current RunPod env | Note |
|--------|---------------------------------------------------|--------------------|------|
| **numpy** | `>=2` (install.sh, requirements.txt) | **1.26.4** | RunPod has 1.x; docs require 2.x. Can cause compatibility issues with JAX/Flax and `numpy>=2`-only code. |
| **torchaudio** | `2.9.0` (install.sh) | **2.2.0** | RunPod has older torchaudio; likely pulled by a dependency. |
| **sentencepiece** | `0.2.1` (requirements.txt) | **0.1.99** | RunPod has older sentencepiece; likely pulled by transformers/tokenizers. |

All other checked packages match (torch 2.9.0+cu128, torchvision 0.24.0, jax 0.5.3, vllm 0.11.2, openpipe-art 0.5.9, langgraph 1.0.8, langchain-core 1.2.12, transformers 4.57.2, trl 0.24.0, unsloth 2026.2.1, flax 0.10.2, mujoco 3.5.0, robosuite 1.4.0, weave 0.52.28, Pillow 12.1.0, scipy 1.17.0, pydantic 2.12.5).

---

## 1. Where versions are specified

- **install.sh**: Python 3.11, conda build deps, `torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0` (cu128), `numpy>=2`, `jax[cuda12]==0.5.3`, `requirements.txt`, `torchcodec>=0.6.0`, LIBERO, patches.
- **INSTALL.md**: Same PyTorch/JAX/numpy versions; `pip install -r requirements.txt`.
- **README.md**: Mentions openpipe-art 0.5.9, vLLM 0.11.x, and manual steps with torch 2.9.0, numpy>=2, jax 0.5.3.
- **requirements.txt**: Pins openpipe-art, vllm, trl, unsloth, transformers, langgraph, langchain-core, numpy>=2, sentencepiece==0.2.1, and many others (see file).

---

## 2. openpi (Pi0 / Pi0.5)

- **Docs (INSTALL.md, README, install.sh verify):** openpi is required for Pi0.5; it can come from:
  - **Option A:** In-repo `agent_attack_framework/openpi/src` (no env var).
  - **Option B:** RoboTwin at `<workspace>/RoboTwin` (or `ROBOTWIN_ROOT`) with `policy/pi05` and the openpi package.
- **Current workspace:** There is no `agent_attack_framework/openpi/` tree in this repo (no `openpi/**/*.py` found). So on RunPod, openpi must be provided by **RoboTwin** (or a similar clone) and `ROBOTWIN_ROOT` (or the default sibling path) must be set so that `libero_rollouts/pi05_libero_model.py` and `pi0_libero_model.py` can `import openpi`.
- **Package version:** The docs do not pin an openpi **pip** version; openpi is used as source (in-repo or RoboTwin). So we are not comparing a “version” of openpi here, only that the RunPod env relies on RoboTwin (or equivalent) for the openpi **code**, not on the repo’s `openpi/` folder.

---

## 3. Recommendations

1. **Align RunPod with docs (recommended):**
   - Recreate the env so numpy and sentencepiece are not downgraded:
     ```bash
     conda activate runpod
     pip install "numpy>=2" "sentencepiece==0.2.1" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/cu128
     ```
   - Or run a full clean install: `bash init_runpod_env.sh` in a fresh env and fix any dependency conflicts (e.g. if a dependency forces numpy<2, consider relaxing that dependency or installing numpy>=2 last).
2. **numpy 1.x vs 2.x:** Code and deps (e.g. JAX, some ML libs) that assume numpy 2.x may break or behave differently with 1.26.4. Aligning to `numpy>=2` is recommended.
3. **torchaudio 2.2 vs 2.9:** Usually only matters if you use audio features; for VLA attack (no audio), impact is low. For strict consistency with install.sh, pin `torchaudio==2.9.0` when reinstalling.
4. **sentencepiece 0.1.99 vs 0.2.1:** Tokenizer behavior can differ slightly; aligning to 0.2.1 keeps you consistent with requirements.txt and avoids subtle tokenization mismatches.

---

## 4. How this was checked

- RunPod: `conda activate runpod` then `pip list` / `pip show` (key packages).
- Docs: `README.md`, `INSTALL.md`, `install.sh`, `requirements.txt`.
- openpi: `agent_attack_framework` has no `openpi/` directory; Pi0/Pi0.5 wrappers expect openpi from in-repo `openpi/src` or RoboTwin.
