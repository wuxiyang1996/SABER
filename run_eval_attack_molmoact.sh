#!/usr/bin/env bash
# Live attack evaluation (task_failure) on MolmoAct.
#
# Uses 3 GPUs per eval (1 VLA + 2 attack agent), running the full attack
# agent pipeline against MolmoAct on LIBERO.
#
# GPU layout (4x A100-80GB):
#   GPU 0: MolmoAct (PyTorch, subprocess via vla_molmoact env)
#   GPU 2: attack agent vLLM inference
#   GPU 3: Unsloth model loading (needed by ART init; idle during inference)
#
# MolmoAct specifics:
#   - Per-suite checkpoints (reloads per suite automatically)
#   - Action horizon = 1 (single-step predictions)
#   - Uses vla_molmoact conda env (subprocess isolation)
#   - Based on Molmo VLM with spatial trace reasoning
#
# Usage:
#   bash run_eval_attack_molmoact.sh                                # default settings
#   bash run_eval_attack_molmoact.sh --vla-gpu 1                    # VLA on GPU 1
#   bash run_eval_attack_molmoact.sh --attack-gpus 2,3              # attack on GPUs 2,3
#   bash run_eval_attack_molmoact.sh --objective constraint_violation  # different objective
#   bash run_eval_attack_molmoact.sh --suites libero_spatial         # single suite
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

# ---- Defaults ----
ATTACK_MODEL="qwen2.5-3B-cold-start"
ATTACK_PROJECT="vla-attack-agent-cold-start"
ATTACK_BASE="outputs/sft_runs/sft_cold_start__Qwen_Qwen2.5-3B-Instruct__20260301_200405__task_failure/merged_model"
OBJECTIVE="task_failure"
TOOL_SETS="token,char,prompt"
SUITES="libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS="7-9"
EPISODES=5
SEED=42
REPLAN_STEPS=1

VLA_GPU=1
ATTACK_GPUS="2,3"
OUTPUT_DIR="outputs/attack_eval_molmoact_task_failure"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vla-gpu)
      VLA_GPU="$2"; shift 2 ;;
    --attack-gpus)
      ATTACK_GPUS="$2"; shift 2 ;;
    --objective)
      OBJECTIVE="$2"; shift 2 ;;
    --suites)
      SUITES="$2"; shift 2 ;;
    --task-ids)
      TASK_IDS="$2"; shift 2 ;;
    --episodes)
      EPISODES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --replan)
      REPLAN_STEPS="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --attack-model)
      ATTACK_MODEL="$2"; shift 2 ;;
    --attack-project)
      ATTACK_PROJECT="$2"; shift 2 ;;
    --attack-base)
      ATTACK_BASE="$2"; shift 2 ;;
    -*)
      echo "Unknown flag: $1"; exit 1 ;;
    *)
      shift ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  MolmoAct Live Attack Evaluation"
echo "  Victim:       molmoact"
echo "  Objective:    ${OBJECTIVE}"
echo "  Attack model: ${ATTACK_MODEL}"
echo "  Suites:       ${SUITES}"
echo "  Task IDs:     ${TASK_IDS}"
echo "  Episodes:     ${EPISODES}"
echo "  Replan:       ${REPLAN_STEPS}"
echo "  GPU layout:   VLA=${VLA_GPU}, Attack=${ATTACK_GPUS}"
echo "  Output:       ${OUTPUT_DIR}/"
echo "========================================"

python eval_attack_vla.py \
  --victim molmoact \
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
  --replan_steps "$REPLAN_STEPS" \
  --max_seq_length 8192 \
  --gpu_memory_utilization 0.7 \
  --rollout_workers 1 \
  2>&1 | tee "${OUTPUT_DIR}/molmoact_attack_eval.log"

echo ""
echo "========================================"
echo "  MolmoAct live attack evaluation complete."
echo "  Report: ${OUTPUT_DIR}/"
echo "========================================"
