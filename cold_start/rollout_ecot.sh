#!/usr/bin/env bash
# Rollout configuration for ECoT (Embodied Chain-of-Thought) as the victim VLA.
#
# Source this file from cold-start collection scripts to set ECoT-specific
# rollout parameters.
#
# Model:       Embodied-CoT/ecot-openvla-7b-bridge
# Framework:   PyTorch / HuggingFace transformers (OpenVLA-based)
# Action dim:  7-DoF (single action per call, action_horizon=1)
# Reasoning:   Embodied CoT; single predict_action call (no long CoT decode in wrapper).
#
# GPU layout:
#   ECoT is 7B; use one GPU per replica. Scale workers with GPU count.
#
# Prerequisites:
#   - No extra repo required; uses HuggingFace trust_remote_code.

# ---- VLA model selection ----
VLA_MODEL="ecot"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

# ---- Rollout parameters ----
REPLAN_STEPS=5

# GPU assignment
VLA_GPUS="${VLA_GPUS:-0,1,2,3}"

# Parallelism
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-32}"
CONCURRENT_LLM="${CONCURRENT_LLM:-32}"

echo "------------------------------------------------------------"
echo "  Rollout config: ECoT (Embodied CoT)"
echo "  Checkpoint:     ${VLA_CHECKPOINT:-Embodied-CoT/ecot-openvla-7b-bridge (default)}"
echo "  Replan steps:   $REPLAN_STEPS"
echo "  VLA GPUs:       $VLA_GPUS"
echo "  Workers:        $ROLLOUT_WORKERS"
echo "------------------------------------------------------------"
