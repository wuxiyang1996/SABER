#!/usr/bin/env bash
# Rollout configuration for DeepThinkVLA as the victim VLA.
#
# Source this file from cold-start collection scripts to set
# DeepThinkVLA-specific rollout parameters.
#
# Model:       yinchenghust/deepthinkvla_libero_cot_rl
# Framework:   PyTorch / HuggingFace transformers
# Action dim:  7-DoF (single action per call, action_horizon=1)
# Reasoning:   Chain-of-thought + RL (generates CoT tokens before actions)
#
# GPU layout:
#   DeepThinkVLA fits on a single GPU (~14 GB in bf16).
#   Multiple GPUs can host separate model replicas for parallel rollouts.
#
# Prerequisites:
#   - DeepThinkVLA repo cloned into repos/deepthinkvla or repos/DeepThinkVLA
#   - HuggingFace checkpoint auto-downloads on first run

# ---- VLA model selection ----
VLA_MODEL="deepthinkvla"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"    # empty = auto-download from HF

# ---- Rollout parameters ----
REPLAN_STEPS=5

# GPU assignment: each GPU hosts one model replica
VLA_GPUS="${VLA_GPUS:-0,1,2,3}"

# Parallelism: scale workers with GPU count
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-32}"
CONCURRENT_LLM="${CONCURRENT_LLM:-32}"

echo "------------------------------------------------------------"
echo "  Rollout config: DeepThinkVLA"
echo "  Checkpoint:     ${VLA_CHECKPOINT:-yinchenghust/deepthinkvla_libero_cot_rl (default)}"
echo "  Replan steps:   $REPLAN_STEPS"
echo "  VLA GPUs:       $VLA_GPUS"
echo "  Workers:        $ROLLOUT_WORKERS"
echo "------------------------------------------------------------"
