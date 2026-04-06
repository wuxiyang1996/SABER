#!/usr/bin/env bash
# Run GRPO training warm-started from a cold-start SFT model.
# Pipeline: cold-start collection -> SFT -> GRPO (this script).
#
# GPUs: VLA rollouts on 0,1; vLLM inference on 2; Unsloth training on 3.
#
# Usage:
#   bash scripts/run_train.sh task_failure              # default objective
#   bash scripts/run_train.sh action_inflation
#   bash scripts/run_train.sh constraint_violation
set -euo pipefail

cd "$(dirname "$0")/.."

OBJECTIVE="${1:?Usage: $0 <task_failure|action_inflation|constraint_violation>}"

# ---- Conda activation ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done
conda activate runpod

# ---- Headless rendering (MuJoCo / robosuite) ----
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# Ensure conda env libs are on the linker path
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Prefer the vast venv Python; fall back to env python
if [[ -x /venv/vast/bin/python ]]; then
  PYTHON=/venv/vast/bin/python
elif command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

# ---- Objective-specific parameters ----
case "$OBJECTIVE" in
  task_failure)
    MODEL_NAME="qwen2.5-3B-cold-start"
    PROJECT_NAME="vla-attack-agent-cold-start"
    BASE_MODEL="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200405__task_failure/merged_model"
    TOOL_SETS="char,token,prompt"
    GROUPS_PER_STEP=2
    MAX_TURNS=8
    EXTRA_ARGS=(--resume)
    ;;
  action_inflation)
    MODEL_NAME="qwen2.5-3B-cold-start-action-inflation"
    PROJECT_NAME="vla-attack-agent-cold-start-action-inflation"
    BASE_MODEL="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200610__action_inflation/merged_model"
    TOOL_SETS="prompt"
    GROUPS_PER_STEP=4
    MAX_TURNS=4
    EXTRA_ARGS=(--no_attack_penalty -1.0 --short_trajectory_penalty 0.2 --short_trajectory_ratio 0.5 --max_steps 800)
    ;;
  constraint_violation)
    MODEL_NAME="qwen2.5-3B-cold-start-constraint-violation"
    PROJECT_NAME="vla-attack-agent-cold-start-constraint-violation"
    BASE_MODEL="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200642__constraint_violation/merged_model"
    TOOL_SETS="prompt"
    GROUPS_PER_STEP=2
    MAX_TURNS=8
    EXTRA_ARGS=(--resume)
    ;;
  *)
    echo "ERROR: Unknown objective '${OBJECTIVE}'. Use: task_failure, action_inflation, constraint_violation"
    exit 1
    ;;
esac

$PYTHON train_vla.py \
  --model_name "$MODEL_NAME" \
  --project_name "$PROJECT_NAME" \
  --base_model "$BASE_MODEL" \
  --objective "$OBJECTIVE" \
  --tool_sets "$TOOL_SETS" \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-9 \
  --episodes_per_task 1 \
  --trajectories_per_group 8 \
  --groups_per_step "$GROUPS_PER_STEP" \
  --num_epochs 8 \
  --learning_rate 5e-05 \
  --stealth_weight 0.03 \
  --max_edit_chars 200 \
  --max_turns "$MAX_TURNS" \
  --replan_steps 5 \
  --seed 42 \
  --rollout_workers 16 \
  --max_seq_length 8192 \
  --lora_r 8 \
  --eval_task_ids 7-9 \
  --eval_episodes_per_task 5 \
  --vla_gpus 0,1 \
  --attack_gpus 2,3 \
  --gpu_memory_utilization "${GPU_MEM_UTIL:-0.35}" \
  "${EXTRA_ARGS[@]}"
