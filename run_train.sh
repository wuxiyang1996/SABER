#!/usr/bin/env bash
# Run VLA attack training (Qwen2.5-3B agent, task_failure objective, token-only).
# GPUs: VLA rollouts on 0,1; vLLM inference on 2; Unsloth training on 3.
# Split-GPU eliminates sleep/wake cycle entirely.
set -euo pipefail

cd "$(dirname "$0")"

# Prefer the vast venv Python; fall back to env python
if [[ -x /venv/vast/bin/python ]]; then
  PYTHON=/venv/vast/bin/python
elif command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

$PYTHON train_vla.py \
  --model_name qwen2.5-3B \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --objective task_failure \
  --tool_sets token \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-9 \
  --episodes_per_task 1 \
  --trajectories_per_group 8 \
  --groups_per_step 2 \
  --num_epochs 10 \
  --learning_rate 5e-05 \
  --stealth_weight 0.03 \
  --max_edit_chars 200 \
  --max_turns 8 \
  --replan_steps 5 \
  --seed 42 \
  --rollout_workers 16 \
  --max_seq_length 8192 \
  --lora_r 16 \
  --eval_task_ids 7-9 \
  --eval_episodes_per_task 5 \
  --vla_gpus 0,1 \
  --attack_gpus 2,3 \
  --gpu_memory_utilization 0.65
