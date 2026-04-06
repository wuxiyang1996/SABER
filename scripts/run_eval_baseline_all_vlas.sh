#!/usr/bin/env bash
# Baseline evaluation of OpenVLA and other VLA models on LIBERO (no attack).
#
# Evaluates each VLA model on the 4 LIBERO suites using eval.run_libero_eval
# (native models) or eval.external.run (external models). No attack agent is
# loaded — this measures clean task-execution performance only.
#
# Outputs are saved under $OUTPUT_DIR:
#   <model>_<timestamp>.json — per-model result JSON (per-suite/task success rates)
#   <model>_<timestamp>.log  — subprocess log
#   all_baseline_metrics_<timestamp>.json — aggregated cross-model performance report
#
# GPU layout (4 GPUs, one model per GPU in parallel):
#   GPU 0: model batch 1   GPU 1: model batch 2
#   GPU 2: model batch 3   GPU 3: model batch 4
#
# Usage:
#   bash scripts/run_eval_baseline_all_vlas.sh                       # all models
#   bash scripts/run_eval_baseline_all_vlas.sh openvla               # single model
#   bash scripts/run_eval_baseline_all_vlas.sh openvla ecot deepthinkvla  # specific models
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
SUITES="libero_spatial,libero_object,libero_goal,libero_10"
TASK_IDS="0-9"
EPISODES=5
SEED=42
REPLAN_STEPS=5
OUTPUT_DIR="outputs/baseline_eval"
GPUS="0,1,2,3"
CPU_WORKERS=4
MODELS_PER_GPU=3

mkdir -p "$OUTPUT_DIR"

# ---- Model list (override with positional args) ----
if [[ $# -gt 0 ]]; then
  MODELS_CSV=$(IFS=,; echo "$*")
else
  MODELS_CSV="openpi_pi05,openvla,deepthinkvla,ecot,molmoact,internvla_m1"
fi

echo "========================================"
echo "  Baseline Evaluation (no attack)"
echo "  Models: ${MODELS_CSV}"
echo "  Suites: ${SUITES}"
echo "  Task IDs: ${TASK_IDS}"
echo "  Episodes/task: ${EPISODES}"
echo "  Seed: ${SEED}"
echo "  GPUs: ${GPUS}"
echo "  Output: ${OUTPUT_DIR}/"
echo "========================================"
echo ""

python -m eval.run_all_libero_evals_parallel \
  --gpus "$GPUS" \
  --models "$MODELS_CSV" \
  --suites "spatial,object,goal,long" \
  --task_ids "$TASK_IDS" \
  --episodes_per_task "$EPISODES" \
  --seed "$SEED" \
  --replan_steps "$REPLAN_STEPS" \
  --cpu_workers "$CPU_WORKERS" \
  --models_per_gpu "$MODELS_PER_GPU" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "${OUTPUT_DIR}/baseline_eval.log"

echo ""
echo "========================================"
echo "  Baseline evaluation complete."
echo "  Reports in: ${OUTPUT_DIR}/"
echo "========================================"
