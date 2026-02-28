#!/usr/bin/env bash
# Run VLA attack training with Qwen2.5-7B-Instruct as the attack agent.
# Usage: ./run_vla_qwen25_7b.sh [objective]
#   objective: task_failure (default), action_inflation, thinking_inflation, hallucination, constraint_violation

set -e
cd "$(dirname "$0")"
OBJECTIVE="${1:-task_failure}"

python run.py vla \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --model_name qwen2.5-7B \
  --objective "$OBJECTIVE" \
  --task_suite libero_spatial \
  --task_ids 0,1,2
