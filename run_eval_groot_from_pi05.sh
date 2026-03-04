#!/usr/bin/env bash
# Evaluate GR00T N1.5-3B (Tacoin fine-tuned) on LIBERO — task failure attack records from openpi_pi05.
#
# Per-suite Tacoin checkpoints:
#   libero_spatial -> Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL-8K
#   libero_object  -> Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT-8K
#   libero_goal    -> Tacoin/GR00T-N1.5-3B-LIBERO-GOAL-8K
#   libero_10      -> Tacoin/GR00T-N1.5-3B-LIBERO-LONG-8K
#
# Usage:
#   bash run_eval_groot_from_pi05.sh                 # GPU 3
#   bash run_eval_groot_from_pi05.sh --gpu 1         # GPU 1
#   bash run_eval_groot_from_pi05.sh --no-aggregate  # skip aggregation
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

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TORCH_HOME=/workspace/.cache_torch
export HF_LEROBOT_HOME=/workspace/.cache/lerobot
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$HF_LEROBOT_HOME"

# ---- Defaults ----
SEED=42
REPLAN_STEPS=8
MAX_STEPS=720
VLA_GPU=3
DO_AGGREGATE=true

VICTIM="groot"
ATTACK_RECORD="outputs/agent_output_records_task_failure_2/task_failure_openpi_pi05.json"
OUTPUT_DIR="outputs/eval_result"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      VLA_GPU="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --replan)
      REPLAN_STEPS="$2"; shift 2 ;;
    --no-aggregate)
      DO_AGGREGATE=false; shift ;;
    -*)
      echo "Unknown flag: $1"; exit 1 ;;
    *)
      shift ;;
  esac
done

if [[ ! -f "$ATTACK_RECORD" ]]; then
  echo "ERROR: Attack record not found: ${ATTACK_RECORD}"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

SOURCE_NAME=$(basename "$ATTACK_RECORD" .json)
LOG_FILE="${OUTPUT_DIR}/${VICTIM}_from_${SOURCE_NAME}.log"

echo "========================================"
echo "  GR00T N1.5 Task Failure Replay Eval"
echo "  Victim:  ${VICTIM}"
echo "  Source:  ${ATTACK_RECORD}"
echo "  GPU:     ${VLA_GPU}"
echo "  Seed:    ${SEED}"
echo "  Replan:  ${REPLAN_STEPS}"
echo "  MaxStep: ${MAX_STEPS}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "========================================"
echo ""

python eval_replay_attack.py \
  --victim "$VICTIM" \
  --vla_gpu "$VLA_GPU" \
  --attack_record "$ATTACK_RECORD" \
  --seed "$SEED" \
  --replan_steps "$REPLAN_STEPS" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "[GPU ${VLA_GPU}] Replay evaluation done: ${VICTIM} <- ${SOURCE_NAME}"

# ---- Aggregate results ----
if [[ "$DO_AGGREGATE" == true ]] && command -v python &>/dev/null; then
  echo ""
  echo "========================================"
  echo "  Aggregating results..."
  echo "========================================"

  python aggregate_replay_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/groot_eval_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  GR00T task failure evaluation complete."
echo "  Report:   ${OUTPUT_DIR}/replay_task_failure_${VICTIM}_from_openpi_pi05.json"
echo "  Summary:  ${OUTPUT_DIR}/groot_eval_summary.json"
echo "========================================"
