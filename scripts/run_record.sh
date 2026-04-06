#!/usr/bin/env bash
# Record all attack-agent outputs given the initial LIBERO instructions.
#
# Runs eval_attack_vla.py against each victim VLA and captures the full agent
# interaction (tool calls, perturbations, baseline/attack rollout metrics) into
# per-model JSON reports and verbose logs.
#
# GPU layout (4x A100-80GB):
#   GPU 0 (VLA) + GPUs 2,3 (attack agent vLLM)
#
# Usage:
#   bash scripts/run_record.sh task_failure                         # all models
#   bash scripts/run_record.sh action_inflation openvla             # single model
#   bash scripts/run_record.sh constraint_violation openvla lightvla
set -euo pipefail

cd "$(dirname "$0")/.."

OBJECTIVE="${1:?Usage: $0 <task_failure|action_inflation|constraint_violation> [model ...]}"
shift

# ---- Conda ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done
conda activate runpod

# ---- Env ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1

# ---- Objective-specific parameters ----
case "$OBJECTIVE" in
  task_failure)
    ATTACK_MODEL="qwen2.5-3B-cold-start"
    ATTACK_PROJECT="vla-attack-agent-cold-start"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200405__task_failure/merged_model"
    TOOL_SETS="token"
    OUTPUT_DIR="outputs/agent_output_records_task_failure_2"
    MAX_TURNS=4
    EXTRA_ARGS=()
    PIN_CKPT_MAX=""
    ;;
  action_inflation)
    ATTACK_MODEL="qwen2.5-3B-cold-start-action-inflation"
    ATTACK_PROJECT="vla-attack-agent-cold-start-action-inflation"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200610__action_inflation/merged_model"
    TOOL_SETS="prompt"
    OUTPUT_DIR="outputs/agent_output_records_action_inflation"
    MAX_TURNS=4
    EXTRA_ARGS=(--no_attack_penalty -1.0 --short_trajectory_penalty 0.2 --short_trajectory_ratio 0.5 --max_steps 800 --select_max_attack_steps)
    PIN_CKPT_MAX="0050"
    ;;
  constraint_violation)
    ATTACK_MODEL="qwen2.5-3B-cold-start-constraint-violation"
    ATTACK_PROJECT="vla-attack-agent-cold-start-constraint-violation"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200642__constraint_violation/merged_model"
    TOOL_SETS="prompt"
    OUTPUT_DIR="outputs/agent_output_records_constraint_violation"
    MAX_TURNS=4
    EXTRA_ARGS=(--no_attack_penalty -1.0 --short_trajectory_penalty 0.2 --short_trajectory_ratio 0.5 --select_max_attack_steps)
    PIN_CKPT_MAX="0150"
    ;;
  *)
    echo "ERROR: Unknown objective '${OBJECTIVE}'."
    exit 1
    ;;
esac

SUITES="libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS="7-9"
EPISODES=5
SEED=42
VLA_GPU=0
ATTACK_GPUS="2,3"

mkdir -p "$OUTPUT_DIR"

# ---- Pin checkpoint: hide newer checkpoints if requested ----
HIDDEN_CKPTS=()
if [[ -n "$PIN_CKPT_MAX" ]]; then
  CKPT_DIR="outputs/${ATTACK_PROJECT}/models/${ATTACK_MODEL}/checkpoints"
  PIN_NUM=$((10#$PIN_CKPT_MAX))
  if [[ -d "$CKPT_DIR" ]]; then
    for ckpt in "$CKPT_DIR"/0*; do
      [[ -d "$ckpt" ]] || continue
      CKPT_NUM=$((10#$(basename "$ckpt")))
      if (( CKPT_NUM > PIN_NUM )); then
        mv "$ckpt" "${ckpt}_hidden"
        HIDDEN_CKPTS+=("${ckpt}_hidden")
        echo "Hidden checkpoint: $(basename "$ckpt")"
      fi
    done
  fi
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
  MODELS=(openpi_pi0 openpi_pi05 openvla lightvla deepthinkvla ecot molmoact internvla_m1)
fi

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
    --max_turns "$MAX_TURNS" \
    --replan_steps 5 \
    --max_seq_length 8192 \
    --gpu_memory_utilization 0.7 \
    --rollout_workers 1 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log_file"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Done recording: ${model}"
}

echo "========================================"
echo "  Record Agent Outputs (${OBJECTIVE})"
echo "  Attack agent: ${ATTACK_MODEL}"
echo "  Models: ${MODELS[*]}"
echo "  Suites: ${SUITES}"
echo "  Task IDs: ${TASK_IDS}"
echo "  Episodes/task: ${EPISODES}"
echo "  Output: ${OUTPUT_DIR}/"
echo "========================================"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=== Evaluating: ${model} ==="
  run_one "$model"
done

echo ""
echo "========================================"
echo "  All recordings complete."
echo "  Per-model JSON reports:"
for model in "${MODELS[@]}"; do
  report="${OUTPUT_DIR}/${OBJECTIVE}_${model}.json"
  if [[ -f "$report" ]]; then
    echo "    + ${report}"
  else
    echo "    - ${report}  (not found — check ${OUTPUT_DIR}/${model}_eval.log)"
  fi
done
echo "========================================"
