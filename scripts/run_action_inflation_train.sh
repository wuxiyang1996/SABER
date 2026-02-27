#!/usr/bin/env bash
# Action-inflation training with updated reward settings:
#   - no_attack_penalty=-1.0 (encourage tool use)
#   - short_trajectory_penalty=0.2 (discourage much shorter post-attack trajectories)
#   - short_trajectory_ratio=0.5 (penalty when attack steps < 50% of baseline)
#
# GPU layout (default): VLA on GPUs 0,1,2; attack agent (vLLM) on GPU 3.
# Override with: --vla_gpus 0,1,2 --attack_gpus 3
#
# Run from repo root or agent_attack_framework:
#   cd /workspace/agent_attack_framework && bash scripts/run_action_inflation_train.sh

set -e
cd "$(dirname "$0")/.."

python train_vla.py \
  --objective action_inflation \
  --tool_sets token,char,prompt \
  --max_turns 8 \
  --stealth_weight 0.3 \
  --no_attack_penalty -1.0 \
  --short_trajectory_penalty 0.2 \
  --short_trajectory_ratio 0.5 \
  --max_edit_chars 200 \
  --max_steps 800 \
  --task_suite libero_spatial,libero_object,libero_goal,libero_10 \
  --task_ids 0-6 \
  --eval_task_ids 7-9 \
  --episodes_per_task 1 \
  --eval_episodes_per_task 5 \
  --seed 42 \
  --replan_steps 5 \
  "$@"
