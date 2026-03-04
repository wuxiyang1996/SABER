#!/usr/bin/env bash
# Run attack evaluation (constraint_violation) on victim VLA models.
# Uses 3 GPUs per eval (1 VLA + 2 attack agent), running models sequentially.
#
# GPU layout (4x A100-80GB):
#   GPU 0: VLA model (JAX or PyTorch)
#   GPU 2: attack agent vLLM inference
#   GPU 3: Unsloth model loading (needed by ART init; idle during inference)
#
# NOTE: 2 attack GPUs are required because ART's register() loads the model
# via Unsloth on one GPU and spawns vLLM on another. With only 1 attack GPU,
# both compete for the same memory → OOM.
#
# Usage:
#   bash run_eval_attack_all_vlas.sh                        # run all 8 models
#   bash run_eval_attack_all_vlas.sh openvla                # run a single model
#   bash run_eval_attack_all_vlas.sh openvla lightvla       # run specific models
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
OUTPUT_DIR="outputs/attack_transfer_eval"

# GPU layout: VLA on GPU 0, attack agent on GPUs 2+3
VLA_GPU=0
ATTACK_GPUS="2,3"

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
    xvla
  )
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
    --stealth_weight 0.03 \
    --max_edit_chars 200 \
    --max_turns 8 \
    --replan_steps 5 \
    --max_seq_length 8192 \
    --gpu_memory_utilization 0.7 \
    --rollout_workers 1 \
    2>&1 | tee "${OUTPUT_DIR}/${model}_eval.log"
  echo "[GPU ${VLA_GPU}+${ATTACK_GPUS}] Done: ${model}"
}

echo "========================================"
echo "  Cross-Model Attack Evaluation (constraint_violation)"
echo "  Attack agent: ${ATTACK_MODEL}"
echo "  Models: ${MODELS[*]}"
echo "  GPU layout: VLA=${VLA_GPU}, Attack=${ATTACK_GPUS} (sequential)"
echo "========================================"

for model in "${MODELS[@]}"; do
  echo ""
  echo "=== Evaluating: ${model} (VLA GPU ${VLA_GPU}, Attack GPUs ${ATTACK_GPUS}) ==="
  run_one "$model"
done

echo ""
echo "========================================"
echo "  All evaluations complete."
echo "  Reports in: ${OUTPUT_DIR}/"
echo "========================================"
