#!/usr/bin/env bash
# Run VLA attack training (Qwen2.5-7B agent, task_failure objective).
# Prereqs: run from repo root with conda/venv activated (e.g. conda activate vast).
# GPUs: VLA rollouts on 0,1,2; attack agent on 3. For 3 GPUs use: --vla_gpus 0,1 --attack_gpus 2
set -euo pipefail

cd "$(dirname "$0")"

# Use python from env; fallback to python3 if python is not available
if command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

$PYTHON train_vla.py \
  --model_name qwen2.5-7B \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --objective task_failure \
  --tool_sets token,char,prompt \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-6 \
  --episodes_per_task 1 \
  --trajectories_per_group 2 \
  --groups_per_step 16 \
  --num_epochs 10 \
  --learning_rate 5e-06 \
  --stealth_weight 0.15 \
  --max_edit_chars 200 \
  --max_turns 8 \
  --replan_steps 5 \
  --max_steps 800 \
  --seed 42 \
  --rollout_workers 24 \
  --eval_task_ids 7-9 \
  --eval_episodes_per_task 5 \
  --vla_gpus 0,1,2 \
  --attack_gpus 3 \
  --gpu_memory_utilization 0.60
