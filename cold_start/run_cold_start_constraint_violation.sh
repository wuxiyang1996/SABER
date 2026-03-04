#!/usr/bin/env bash
# Cold-start trajectory collection: GPT-5 Mini attacks Pi0.5 in LIBERO.
#
# Objective: CONSTRAINT VIOLATION — prompt the robot to reach its physical
# limits (joint saturation, collisions, excessive contact forces, out-of-
# range actions).
#
# Collects successful attack trajectories WITHOUT any GRPO training.
# GPT-5 Mini calls the same tool_sets (token/char/prompt) used by the
# local Qwen attack agent, and trajectories with reward >= threshold
# are saved for later warm-start fine-tuning.
#
# Key differences from task_failure / action_inflation shells:
#   --objective constraint_violation
#   --stealth_weight 0.02   (lower: we care more about causing violations
#                            than about perturbation minimality)
#   --reward_threshold 0.2  (lower: constraint violations are harder to
#                            trigger, so accept weaker signals early)
#   --max_steps 1000        (longer horizon: give the VLA room to
#                            accumulate sustained joint-limit / force events)
#   --max_edit_chars 300    (slightly larger budget: constraint-inducing
#                            prompts often need spatial/motion clauses)
#
# Settings mirror run_train.sh:
#   --rollout_workers 32    (same as training)
#   --num_epochs 50         (50 full passes over all 40 scenarios = 2000 episodes)
#   --task_ids 0-9          (all 10 tasks per suite, 4 suites = 40 scenarios)
#   --episodes_per_task 1   (1 initial state per task)
#   --max_turns 8           (8 ReAct tool-call rounds per episode)
#   --replan_steps 5        (VLA inference every 5 env steps)
#
# Total episodes = 4 suites x 10 tasks x 1 ep/task x 50 epochs = 2000
#
# Prerequisites:
#   - OPENAI_API_KEY set in environment or .env
#   - Pi0.5 checkpoint available (auto-downloads from GCS)
#   - LIBERO installed
#
# GPU layout:
#   GPU 0,1     — Pi0.5 VLA model (JAX)
#   No attack GPU needed — GPT-5 Mini runs via API
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- Conda activation ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
    source "${p}/etc/profile.d/conda.sh"
    break
  fi
done
conda activate runpod 2>/dev/null || conda activate vast 2>/dev/null || true

# ---- Headless rendering (MuJoCo / robosuite) ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1

# Prefer the vast venv Python; fall back to env python
if [[ -x /venv/vast/bin/python ]]; then
  PYTHON=/venv/vast/bin/python
elif command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

# ---- Load .env if present (so OPENAI_API_KEY is available in bash) ----
if [[ -f "$(dirname "$0")/../.env" ]]; then
  set -a
  source "$(dirname "$0")/../.env"
  set +a
fi

echo "============================================================"
echo "Cold-Start Trajectory Collection — GPT-5 Mini → Pi0.5"
echo "  Objective: constraint_violation"
echo "============================================================"
echo "Python:  $PYTHON"
echo "API key: ${OPENAI_API_KEY:+set (${#OPENAI_API_KEY} chars, ...${OPENAI_API_KEY: -4})}${OPENAI_API_KEY:-NOT SET}"
echo "============================================================"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it first:"
  echo "  export OPENAI_API_KEY=sk-..."
  exit 1
fi

$PYTHON -m cold_start.collect \
  --objective constraint_violation \
  --tool_sets token,char,prompt \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-9 \
  --episodes_per_task 1 \
  --num_epochs 50 \
  --max_turns 8 \
  --max_edit_chars 300 \
  --replan_steps 5 \
  --max_steps 1000 \
  --seed 42 \
  --stealth_weight 0.02 \
  --reward_threshold 0.2 \
  --vla_gpus 0,1,2,3 \
  --rollout_workers 32 \
  --concurrent_llm 32 \
  "$@"
