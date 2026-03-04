#!/usr/bin/env bash
# Evaluate SmolVLA on LIBERO using pre-recorded attack prompts from openpi_pi05.
#
# Reads original_instruction / perturbed_instruction from the openpi_pi05 attack
# record, cross-checks original_instruction against LIBERO ground truth, and runs
# SmolVLA twice per episode (baseline + attack).
#
# Output goes to outputs/eval_result/ in the same JSON format as the recorded replay.
#
# SmolVLA specifics:
#   - 450M param VLA (SmolVLM2-500M backbone)
#   - LeRobot framework (lerobot >= 0.4)
#   - Input: 2 images (256x256) + 8-dim state + language
#   - Output: 7-dim action (1 action per call)
#   - Single checkpoint: HuggingFaceVLA/smolvla_libero (all suites)
#   - Runs in vla_smolvla conda env (subprocess isolation)
#
# Usage:
#   bash run_eval_smolvla_from_pi05.sh                 # GPU 3
#   bash run_eval_smolvla_from_pi05.sh --gpu 1         # GPU 1
#   bash run_eval_smolvla_from_pi05.sh --no-aggregate  # skip aggregation
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
SEED=42
REPLAN_STEPS=5
VLA_GPU=3
DO_AGGREGATE=true

VICTIM="smolvla"
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
echo "  SmolVLA Replay Attack Evaluation"
echo "  Victim:  ${VICTIM}"
echo "  HF:      HuggingFaceVLA/smolvla_libero"
echo "  Source:  ${ATTACK_RECORD}"
echo "  GPU:     ${VLA_GPU}"
echo "  Seed:    ${SEED}"
echo "  Replan:  ${REPLAN_STEPS}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "========================================"
echo ""

python eval_replay_attack.py \
  --victim "$VICTIM" \
  --vla_gpu "$VLA_GPU" \
  --attack_record "$ATTACK_RECORD" \
  --seed "$SEED" \
  --replan_steps "$REPLAN_STEPS" \
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
    --output "$OUTPUT_DIR/smolvla_eval_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  SmolVLA evaluation complete."
echo "  Report:   ${OUTPUT_DIR}/replay_task_failure_${VICTIM}_from_openpi_pi05.json"
echo "  Summary:  ${OUTPUT_DIR}/smolvla_eval_summary.json"
echo "========================================"
