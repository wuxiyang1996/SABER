#!/usr/bin/env bash
# Evaluate X-VLA on LIBERO using pre-recorded constraint-violation attack prompts from openpi_pi05.
#
# Reads original_instruction / perturbed_instruction from the openpi_pi05 constraint
# violation attack record, cross-checks original_instruction against LIBERO ground
# truth, and runs X-VLA twice per episode (baseline + attack).
#
# Output goes to outputs/eval_result/ in the same JSON format as the recorded replay.
#
# X-VLA specifics:
#   - Uses absolute EE actions (controller.use_delta = False)
#   - Needs predict_from_obs (raw EE pose for proprioception)
#   - Runs in vla_xvla conda env (subprocess isolation)
#   - Single checkpoint: 2toINF/X-VLA-Libero (all suites)
#
# Usage:
#   bash run_eval_xvla_from_pi05_constraint_violation.sh                 # GPU 0
#   bash run_eval_xvla_from_pi05_constraint_violation.sh --gpu 2         # GPU 2
#   bash run_eval_xvla_from_pi05_constraint_violation.sh --no-aggregate  # skip aggregation
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
VLA_GPU=0
DO_AGGREGATE=true

VICTIM="xvla"
ATTACK_RECORD="outputs/agent_output_records_constraint_violation/constraint_violation_openpi_pi05.json"
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
echo "  X-VLA Constraint Violation Replay Attack Evaluation"
echo "  Victim:  ${VICTIM} (2toINF/X-VLA-Libero)"
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
    --output "$OUTPUT_DIR/xvla_eval_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  X-VLA constraint violation evaluation complete."
echo "  Report:   ${OUTPUT_DIR}/replay_constraint_violation_${VICTIM}_from_openpi_pi05.json"
echo "  Summary:  ${OUTPUT_DIR}/xvla_eval_summary.json"
echo "========================================"
