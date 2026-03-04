#!/usr/bin/env bash
# Record all attack-agent outputs given the initial LIBERO instructions (constraint_violation objective).
#
# This script runs eval_attack_vla.py against each victim VLA and captures
# the full agent interaction (tool calls, perturbations, baseline/attack rollout
# metrics) into per-model JSON reports and verbose logs.
#
# Each task runs 8 trials; only the trial with the most constraint violations
# is kept for reporting (best-of-8 selection via --select_max_attack_steps).
#
# Outputs are saved under $OUTPUT_DIR:
#   <model>_eval.log                        — full stdout/stderr log
#   constraint_violation_<model>.json       — structured JSON with per-episode records including
#                                             original_instruction, perturbed_instruction,
#                                             tools_used, baseline & attack success/steps,
#                                             constraint_violations, collisions, joint_limit_violations,
#                                             excessive_force, reward
#
# GPU layout (4x A100-80GB):
#   Thread A: GPU 0 (VLA) + GPU 1 (attack agent vLLM)
#   Thread B: GPU 2 (VLA) + GPU 3 (attack agent vLLM)
#
# Usage:
#   bash run_record_agent_outputs_constraint_violation.sh                        # all 8 models
#   bash run_record_agent_outputs_constraint_violation.sh openvla                # single model
#   bash run_record_agent_outputs_constraint_violation.sh openvla lightvla       # two in parallel
set -euo pipefail

cd "$(dirname "$0")"

# ---- Conda ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done
conda activate runpod

# ---- Env ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1

# ---- Common args ----
ATTACK_MODEL="qwen2.5-3B-cold-start-constraint-violation"
ATTACK_PROJECT="vla-attack-agent-cold-start-constraint-violation"
ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200642__constraint_violation/merged_model"
OBJECTIVE="constraint_violation"
TOOL_SETS="prompt"
SUITES="libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS="7-9"
EPISODES=5
SEED=42
ROLLOUT_WORKERS=1
OUTPUT_DIR="outputs/agent_output_records_constraint_violation"

mkdir -p "$OUTPUT_DIR"

# ---- Pin checkpoint to 0150: hide any newer checkpoints ----
CKPT_DIR="outputs/vla-attack-agent-cold-start-constraint-violation/models/qwen2.5-3B-cold-start-constraint-violation/checkpoints"
HIDDEN_CKPTS=()
if [[ -d "$CKPT_DIR" ]]; then
  for ckpt in "$CKPT_DIR"/0{151..999}; do
    [[ -d "$ckpt" ]] || continue
    mv "$ckpt" "${ckpt}_hidden"
    HIDDEN_CKPTS+=("${ckpt}_hidden")
    echo "Hidden checkpoint: $(basename "$ckpt") → $(basename "${ckpt}_hidden")"
  done
fi

restore_checkpoints() {
  for h in "${HIDDEN_CKPTS[@]}"; do
    [[ -d "$h" ]] && mv "$h" "${h%_hidden}" && echo "Restored: $(basename "${h%_hidden}")"
  done
}
trap restore_checkpoints EXIT

# ---- Model list (override with positional args) ----
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    openpi_pi0
    openpi_pi05
    openvla
    lightvla
    deepthinkvla
    ecot
    molmoact
    internvla_m1
  )
fi

VLA_GPU=0
ATTACK_GPUS="2,3"

run_one() {
  local model=$1
  local log_file="${OUTPUT_DIR}/${model}_eval.log"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Recording agent outputs — victim: ${model}"
  python eval_attack_vla.py \
    --victim "$model" \
    --vla_gpu "$VLA_GPU" \
    --attack_gpus "$ATTACK_GPUS" \
    --attack_model_name "$ATTACK_MODEL" \
    --attack_project "$ATTACK_PROJECT" \
    --attack_base_model "$ATTACK_BASE" \
    --objective "$OBJECTIVE" \
    --tool_sets "$TOOL_SETS" \
    --suites "$SUITES" \
    --task_ids "$TASK_IDS" \
    --episodes_per_task "$EPISODES" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --stealth_weight 0.03 \
    --no_attack_penalty -1.0 \
    --short_trajectory_penalty 0.2 \
    --short_trajectory_ratio 0.5 \
    --max_edit_chars 200 \
    --max_turns 4 \
    --replan_steps 5 \
    --max_seq_length 8192 \
    --gpu_memory_utilization 0.7 \
    --select_max_attack_steps \
    --rollout_workers "$ROLLOUT_WORKERS" \
    2>&1 | tee "$log_file"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Done recording: ${model}"
}

echo "========================================"
echo "  Record Agent Outputs (constraint_violation)"
echo "  Attack agent: ${ATTACK_MODEL}"
echo "  Models: ${MODELS[*]}"
echo "  Suites: ${SUITES}"
echo "  Task IDs: ${TASK_IDS} (all tasks)"
echo "  Episodes/task: ${EPISODES}"
echo "  Output: ${OUTPUT_DIR}/"
echo "  Sequential (VLA GPU ${VLA_GPU}, Attack GPUs ${ATTACK_GPUS})"
echo "========================================"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=== Evaluating: ${model} (VLA GPU ${VLA_GPU}, Attack GPUs ${ATTACK_GPUS}) ==="
  run_one "$model"
done

# ---- Summarise recorded outputs ----
echo ""
echo "========================================"
echo "  All recordings complete."
echo "  Per-model JSON reports:"
for model in "${MODELS[@]}"; do
  report="${OUTPUT_DIR}/constraint_violation_${model}.json"
  if [[ -f "$report" ]]; then
    echo "    ✓ ${report}"
  else
    echo "    ✗ ${report}  (not found — check ${OUTPUT_DIR}/${model}_eval.log)"
  fi
done
echo ""
echo "  Full logs:   ${OUTPUT_DIR}/*_eval.log"
echo "========================================"
