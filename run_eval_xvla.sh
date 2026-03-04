#!/usr/bin/env bash
# Evaluate X-VLA on LIBERO using pre-recorded attack and non-attack prompts.
#
# Runs on GPU 2. Uses the replay evaluation path (eval_replay_attack.py) which
# executes X-VLA twice per episode:
#   1. Baseline rollout with the original (non-attack) instruction
#   2. Attack rollout with the perturbed instruction
#
# This measures X-VLA's robustness to adversarial prompt perturbations
# without running the attack agent itself.
#
# X-VLA specifics:
#   - Uses absolute EE actions (controller.use_delta = False)
#   - Needs predict_from_obs (raw EE pose for proprioception)
#   - Runs in vla_xvla conda env (subprocess isolation)
#   - Single checkpoint: 2toINF/X-VLA-Libero (all suites)
#
# Sources: pre-recorded attack records from openpi_pi0 and openpi_pi05
# Objectives: task_failure (from agent_output_records_task_failure_2)
#             constraint_violation (from attack_transfer_eval)
#
# Usage:
#   bash run_eval_xvla.sh                      # run all available records
#   bash run_eval_xvla.sh --source pi05        # only pi05-sourced attacks
#   bash run_eval_xvla.sh --source pi0         # only pi0-sourced attacks
#   bash run_eval_xvla.sh --gpu 3              # override GPU (default: 2)
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
VLA_GPU=2
SOURCE_FILTER=""
DO_AGGREGATE=true

OUTPUT_DIR="outputs/replay_eval_xvla"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      VLA_GPU="$2"; shift 2 ;;
    --source)
      SOURCE_FILTER="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_DIR"

# ---- Collect attack records ----
# Task failure records
TF_RECORD_DIR="outputs/agent_output_records_task_failure_2"
TF_RECORDS=()
case "${SOURCE_FILTER,,}" in
  pi0)
    [[ -f "${TF_RECORD_DIR}/task_failure_openpi_pi0.json" ]] && \
      TF_RECORDS+=("${TF_RECORD_DIR}/task_failure_openpi_pi0.json") ;;
  pi05|pi0.5)
    [[ -f "${TF_RECORD_DIR}/task_failure_openpi_pi05.json" ]] && \
      TF_RECORDS+=("${TF_RECORD_DIR}/task_failure_openpi_pi05.json") ;;
  ""|both|all)
    [[ -f "${TF_RECORD_DIR}/task_failure_openpi_pi0.json" ]] && \
      TF_RECORDS+=("${TF_RECORD_DIR}/task_failure_openpi_pi0.json")
    [[ -f "${TF_RECORD_DIR}/task_failure_openpi_pi05.json" ]] && \
      TF_RECORDS+=("${TF_RECORD_DIR}/task_failure_openpi_pi05.json") ;;
esac

# Constraint violation records
CV_RECORD_DIR="outputs/attack_transfer_eval"
CV_RECORDS=()
case "${SOURCE_FILTER,,}" in
  pi0)
    [[ -f "${CV_RECORD_DIR}/constraint_violation_openpi_pi0.json" ]] && \
      CV_RECORDS+=("${CV_RECORD_DIR}/constraint_violation_openpi_pi0.json") ;;
  pi05|pi0.5)
    [[ -f "${CV_RECORD_DIR}/constraint_violation_openpi_pi05.json" ]] && \
      CV_RECORDS+=("${CV_RECORD_DIR}/constraint_violation_openpi_pi05.json") ;;
  ""|both|all)
    [[ -f "${CV_RECORD_DIR}/constraint_violation_openpi_pi0.json" ]] && \
      CV_RECORDS+=("${CV_RECORD_DIR}/constraint_violation_openpi_pi0.json")
    [[ -f "${CV_RECORD_DIR}/constraint_violation_openpi_pi05.json" ]] && \
      CV_RECORDS+=("${CV_RECORD_DIR}/constraint_violation_openpi_pi05.json") ;;
esac

ALL_RECORDS=("${TF_RECORDS[@]}" "${CV_RECORDS[@]}")

if [[ ${#ALL_RECORDS[@]} -eq 0 ]]; then
  echo "ERROR: No attack records found."
  echo "  Checked: ${TF_RECORD_DIR}/ and ${CV_RECORD_DIR}/"
  exit 1
fi

echo "========================================"
echo "  X-VLA Replay Attack Evaluation"
echo "  Victim:   xvla (2toINF/X-VLA-Libero)"
echo "  GPU:      ${VLA_GPU}"
echo "  Records:  ${#ALL_RECORDS[@]} files"
echo "  Replan:   ${REPLAN_STEPS}"
echo "  Output:   ${OUTPUT_DIR}/"
echo "========================================"

run_one() {
  local record=$1
  local source_name
  source_name=$(basename "$record" .json)
  local log_file="${OUTPUT_DIR}/xvla_from_${source_name}.log"

  echo "[GPU ${VLA_GPU}] Replaying: ${source_name} -> xvla"

  python eval_replay_attack.py \
    --victim xvla \
    --vla_gpu "$VLA_GPU" \
    --attack_record "$record" \
    --seed "$SEED" \
    --replan_steps "$REPLAN_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$log_file"

  echo "[GPU ${VLA_GPU}] Done: ${source_name}"
}

for record in "${ALL_RECORDS[@]}"; do
  echo ""
  echo "=== xvla <- $(basename "$record") ==="
  run_one "$record"
done

# ---- Aggregate ----
if [[ "$DO_AGGREGATE" == true ]] && command -v python &>/dev/null; then
  echo ""
  echo "========================================"
  echo "  Aggregating results..."
  echo "========================================"

  python aggregate_replay_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/xvla_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  X-VLA evaluation complete."
echo "  Reports:  ${OUTPUT_DIR}/replay_*.json"
echo "  Summary:  ${OUTPUT_DIR}/xvla_summary.json"
echo "========================================"
