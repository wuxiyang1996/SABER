#!/usr/bin/env bash
# Cold-start trajectory collection: GPT-5 Mini attacks DeepThinkVLA in LIBERO.
#
# Objective: HALLUCINATION — induce the VLA to hallucinate objects, spatial
# relationships, or action semantics that don't match the scene, while the
# VLA still completes the task.  Reward uses four sub-signals: predicate
# mismatch, action jerk, LLM judge, and perturbation leakage.
#
# DeepThinkVLA's chain-of-thought reasoning exposes a rich attack surface
# for hallucination: its reasoning text can be evaluated for factual
# grounding against the actual scene state.
#
# Key parameters (vs default task_failure):
#   --objective hallucination
#   --stealth_weight 0.02   (lower: prioritise triggering hallucinations
#                            over perturbation minimality)
#   --reward_threshold 0.2  (lower: hallucinations are harder to trigger,
#                            accept weaker signals early)
#   --max_edit_chars 300    (larger budget: hallucination-inducing prompts
#                            often need descriptive spatial clauses)
#   --max_steps 1000        (longer horizon: give VLA room to accumulate
#                            predicate mismatches across steps)
#   --replan_steps 5        (DeepThinkVLA)
#
# Total episodes = 4 suites x 10 tasks x 1 ep/task x 50 epochs = 2000
#
# Prerequisites:
#   - OPENAI_API_KEY set in environment or .env
#   - DeepThinkVLA checkpoint (auto-downloads from HuggingFace)
#   - DeepThinkVLA repo in repos/deepthinkvla or repos/DeepThinkVLA
#   - LIBERO installed
#
# GPU layout:
#   VLA_GPUS — DeepThinkVLA (PyTorch, ~14 GB each in bf16)
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

# ---- Load DeepThinkVLA rollout config ----
source "$(dirname "$0")/rollout_deepthinkvla.sh"

echo "============================================================"
echo "Cold-Start Trajectory Collection — GPT-5 Mini → DeepThinkVLA"
echo "  Objective: hallucination"
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
  --vla_model "$VLA_MODEL" \
  ${VLA_CHECKPOINT:+--vla_checkpoint "$VLA_CHECKPOINT"} \
  --objective hallucination \
  --tool_sets token,char,prompt \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-9 \
  --episodes_per_task 1 \
  --num_epochs 50 \
  --max_turns 8 \
  --max_edit_chars 300 \
  --replan_steps "$REPLAN_STEPS" \
  --max_steps 1000 \
  --seed 42 \
  --stealth_weight 0.02 \
  --reward_threshold 0.2 \
  --vla_gpus "$VLA_GPUS" \
  --rollout_workers "$ROLLOUT_WORKERS" \
  --concurrent_llm "$CONCURRENT_LLM" \
  "$@"
