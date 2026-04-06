#!/usr/bin/env bash
# Run live attack evaluation on victim VLA models.
# Uses 3 GPUs per eval (1 VLA + 2 attack agent), running models sequentially.
#
# GPU layout (4x A100-80GB):
#   GPU 0: VLA model (JAX or PyTorch)
#   GPU 2,3: attack agent vLLM + Unsloth
#
# Usage:
#   bash scripts/run_eval_attack.sh task_failure                       # all models
#   bash scripts/run_eval_attack.sh constraint_violation openvla       # single model
#   bash scripts/run_eval_attack.sh action_inflation openvla lightvla  # specific models
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

# NOTE: Per-model conda environments are handled automatically by Python code.
# model_factory.py launches non-JAX VLA models in subprocesses using
# SubprocessVLAWrapper, which picks the correct conda env's Python binary
# via _MODEL_ENV_MAP. See scripts/setup_vla_envs.sh for env creation.

# ---- Objective-specific parameters ----
case "$OBJECTIVE" in
  task_failure)
    ATTACK_MODEL="qwen2.5-3B-cold-start"
    ATTACK_PROJECT="vla-attack-agent-cold-start"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200405__task_failure/merged_model"
    TOOL_SETS="token,char,prompt"
    MAX_TURNS=8
    OUTPUT_DIR="outputs/attack_transfer_eval_task_failure"
    EXTRA_ARGS=()
    ;;
  action_inflation)
    ATTACK_MODEL="qwen2.5-3B-cold-start-action-inflation"
    ATTACK_PROJECT="vla-attack-agent-cold-start-action-inflation"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200610__action_inflation/merged_model"
    TOOL_SETS="token"
    MAX_TURNS=4
    OUTPUT_DIR="outputs/attack_transfer_eval_action_inflation"
    EXTRA_ARGS=(--stealth_weight 0.3 --no_attack_penalty -1.0 --short_trajectory_penalty 0.2 --short_trajectory_ratio 0.5 --max_steps 800 --select_max_attack_steps)
    ;;
  constraint_violation)
    ATTACK_MODEL="qwen2.5-3B-cold-start-constraint-violation"
    ATTACK_PROJECT="vla-attack-agent-cold-start-constraint-violation"
    ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200642__constraint_violation/merged_model"
    TOOL_SETS="prompt"
    MAX_TURNS=8
    OUTPUT_DIR="outputs/attack_transfer_eval"
    EXTRA_ARGS=()
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
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"

mkdir -p "$OUTPUT_DIR"

# ---- Model list (override with positional args) ----
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(openpi_pi0 openpi_pi05 openvla lightvla deepthinkvla ecot molmoact internvla_m1 xvla)
fi

run_one() {
  local model=$1
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Evaluating victim: ${model}"
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
    --stealth_weight "${STEALTH_WEIGHT:-0.03}" \
    --max_edit_chars 200 \
    --max_turns "$MAX_TURNS" \
    --replan_steps 5 \
    --max_seq_length 8192 \
    --gpu_memory_utilization 0.7 \
    --rollout_workers "$ROLLOUT_WORKERS" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/${model}_eval.log"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Done: ${model}"
}

echo "========================================"
echo "  Cross-Model Attack Evaluation (${OBJECTIVE})"
echo "  Attack agent: ${ATTACK_MODEL}"
echo "  Models: ${MODELS[*]}"
echo "  GPU layout: VLA=${VLA_GPU}, Attack=${ATTACK_GPUS} (sequential)"
echo "========================================"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=== Evaluating: ${model} ==="
  run_one "$model"
done

echo ""
echo "========================================"
echo "  All evaluations complete."
echo "  Reports in: ${OUTPUT_DIR}/"
echo "========================================"
