#!/usr/bin/env bash
# Record all attack-agent outputs given the initial LIBERO instructions (task_failure objective).
#
# This script runs eval_attack_vla.py against each victim VLA and captures
# the full agent interaction (tool calls, perturbations, baseline/attack rollout
# metrics) into per-model JSON reports and verbose logs.
#
# Outputs are saved under $OUTPUT_DIR:
#   <model>_eval.log          — full stdout/stderr log
#   task_failure_<model>.json — structured JSON with per-episode records including
#                               original_instruction, perturbed_instruction,
#                               tools_used, baseline & attack success/steps, reward
#
# GPU layout (4x A100-80GB):
#   Thread A: GPU 0 (VLA) + GPU 1 (attack agent vLLM)
#   Thread B: GPU 2 (VLA) + GPU 3 (attack agent vLLM)
#
# Usage:
#   bash run_record_agent_outputs_task_failure.sh                        # all 8 models
#   bash run_record_agent_outputs_task_failure.sh openvla                # single model
#   bash run_record_agent_outputs_task_failure.sh openvla lightvla       # two in parallel
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
ATTACK_MODEL="qwen2.5-3B-cold-start"
ATTACK_PROJECT="vla-attack-agent-cold-start"
ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200405__task_failure/merged_model"
OBJECTIVE="task_failure"
TOOL_SETS="token"
SUITES="libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS="7-9"
EPISODES=5
SEED=42
OUTPUT_DIR="outputs/agent_output_records_task_failure_2"

mkdir -p "$OUTPUT_DIR"

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
    --max_edit_chars 200 \
    --max_turns 4 \
    --replan_steps 5 \
    --max_seq_length 8192 \
    --gpu_memory_utilization 0.7 \
    --rollout_workers 1 \
    2>&1 | tee "$log_file"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Done recording: ${model}"
}

echo "========================================"
echo "  Record Agent Outputs (task_failure)"
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
  report="${OUTPUT_DIR}/task_failure_${model}.json"
  if [[ -f "$report" ]]; then
    echo "    ✓ ${report}"
  else
    echo "    ✗ ${report}  (not found — check ${OUTPUT_DIR}/${model}_eval.log)"
  fi
done
echo ""
echo "  Full logs:   ${OUTPUT_DIR}/*_eval.log"
echo "========================================"
