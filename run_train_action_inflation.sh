#!/usr/bin/env bash
# Run GRPO training (action_inflation) warm-started from a cold-start SFT model.
# Pipeline: cold-start collection → SFT → GRPO (this script).
#
# Base model: cold-start SFT LoRA adapter (Qwen2.5-3B-Instruct + action_inflation trajectories).
# GPUs: VLA rollouts on 0,1; vLLM inference on 2; Unsloth training on 3.
#
# ~6h target: Total time ≈ (steps × time_per_step) + post_eval.
#   steps = (num_scenarios / groups_per_step) × num_epochs.
#   With task_ids 0-9, 4 suites, 1 ep/task → 40 scenarios → 10 steps/epoch.
#   Presets for ~6h (tune after measuring time_per_step):
#     • num_epochs 5 → 50 steps (if ~6–7 min/step → ~5.5h train + ~30m eval).
#     • Shorter: num_epochs 4. Longer: num_epochs 8 or 10.
#     • Much shorter steps: --groups_per_step 2 --trajectories_per_group 4 (noisier gradient).
set -euo pipefail

# ---- Headless rendering (MuJoCo / robosuite) — set before any Python/conda so child processes inherit ----
# If EGL fails (e.g. AttributeError: 'NoneType' object has no attribute 'eglQueryString'), install:
#   conda install -c conda-forge libopengl mesalib
# or use OSMesa:  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUTF8=1

cd "$(dirname "$0")"

# ---- Conda activation ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
    source "${p}/etc/profile.d/conda.sh"
    break
  fi
done
conda activate runpod

# Ensure conda env libs (libEGL, libOpenGL, etc.) are on the linker path
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Reduce OOM: expandable_segments reduces fragmentation; lower util leaves vLLM headroom
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# Prefer the vast venv Python; fall back to env python
if [[ -x /venv/vast/bin/python ]]; then
  PYTHON=/venv/vast/bin/python
elif command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

$PYTHON train_vla.py \
  --model_name qwen2.5-3B-cold-start-action-inflation \
  --project_name vla-attack-agent-cold-start-action-inflation \
  --base_model outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200610__action_inflation/merged_model \
  --objective action_inflation \
  --tool_sets prompt \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-9 \
  --episodes_per_task 1 \
  --trajectories_per_group 8 \
  --groups_per_step 4 \
  --num_epochs 8 \
  --learning_rate 5e-05 \
  --stealth_weight 0.03 \
  --no_attack_penalty -1.0 \
  --short_trajectory_penalty 0.2 \
  --short_trajectory_ratio 0.5 \
  --max_edit_chars 200 \
  --max_steps 800 \
  --max_turns 4 \
  --replan_steps 5 \
  --seed 42 \
  --rollout_workers 16 \
  --max_seq_length 8192 \
  --lora_r 8 \
  --eval_task_ids 7-9 \
  --eval_episodes_per_task 5 \
  --vla_gpus 0,1 \
  --attack_gpus 2,3 \
  --gpu_memory_utilization 0.35
