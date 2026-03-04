#!/usr/bin/env bash
# Evaluate MolmoAct on LIBERO using pre-recorded attack and non-attack prompts.
#
# Runs on a single GPU. Uses the replay evaluation path (eval_replay_attack.py)
# which executes MolmoAct twice per episode:
#   1. Baseline rollout with the original (non-attack) instruction
#   2. Attack rollout with the perturbed instruction
#
# MolmoAct specifics:
#   - Per-suite checkpoints (allenai/MolmoAct-7B-D-LIBERO-{Spatial,Object,Goal,Long}-0812)
#   - Standard delta actions (use_delta = True)
#   - Action horizon = 1 (single-step predictions)
#   - Runs in vla_molmoact conda env (subprocess isolation, transformers >= 4.51)
#   - Based on the Molmo vision-language model with spatial trace reasoning
#
# Sources: pre-recorded attack records from openpi_pi0 and openpi_pi05
# Objectives: task_failure (from agent_output_records_task_failure_2)
#             constraint_violation (from attack_transfer_eval)
#
# Usage:
#   bash run_eval_molmoact.sh                        # run all available records
#   bash run_eval_molmoact.sh --source pi05          # only pi05-sourced attacks
#   bash run_eval_molmoact.sh --source pi0           # only pi0-sourced attacks
#   bash run_eval_molmoact.sh --gpu 1                # override GPU (default: 0)
#   bash run_eval_molmoact.sh --tf-only              # task_failure records only
#   bash run_eval_molmoact.sh --cv-only              # constraint_violation records only
#   bash run_eval_molmoact.sh --no-aggregate         # skip aggregation step
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
REPLAN_STEPS=1
VLA_GPU=0
SOURCE_FILTER=""
DO_AGGREGATE=true
INCLUDE_TF=true
INCLUDE_CV=true

OUTPUT_DIR="outputs/replay_eval_molmoact"

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
    --tf-only)
      INCLUDE_CV=false; shift ;;
    --cv-only)
      INCLUDE_TF=false; shift ;;
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
if [[ "$INCLUDE_TF" == true ]]; then
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
fi

# Constraint violation records
CV_RECORD_DIR="outputs/attack_transfer_eval"
CV_RECORDS=()
if [[ "$INCLUDE_CV" == true ]]; then
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
fi

ALL_RECORDS=("${TF_RECORDS[@]}" "${CV_RECORDS[@]}")

if [[ ${#ALL_RECORDS[@]} -eq 0 ]]; then
  echo "ERROR: No attack records found."
  echo "  Checked: ${TF_RECORD_DIR}/ and ${CV_RECORD_DIR}/"
  echo "  Source filter: '${SOURCE_FILTER}'"
  echo "  Include task_failure: ${INCLUDE_TF}"
  echo "  Include constraint_violation: ${INCLUDE_CV}"
  exit 1
fi

echo "========================================"
echo "  MolmoAct Replay Attack Evaluation"
echo "  Victim:   molmoact"
echo "  Checkpoints:"
echo "    Spatial: allenai/MolmoAct-7B-D-LIBERO-Spatial-0812"
echo "    Object:  allenai/MolmoAct-7B-D-LIBERO-Object-0812"
echo "    Goal:    allenai/MolmoAct-7B-D-LIBERO-Goal-0812"
echo "    Long:    allenai/MolmoAct-7B-D-LIBERO-Long-0812"
echo "  Env:      vla_molmoact"
echo "  GPU:      ${VLA_GPU}"
echo "  Records:  ${#ALL_RECORDS[@]} files"
echo "  Replan:   ${REPLAN_STEPS}"
echo "  Output:   ${OUTPUT_DIR}/"
echo "========================================"

run_one() {
  local record=$1
  local source_name
  source_name=$(basename "$record" .json)
  local log_file="${OUTPUT_DIR}/molmoact_from_${source_name}.log"

  echo "[GPU ${VLA_GPU}] Replaying: ${source_name} -> molmoact"

  python eval_replay_attack.py \
    --victim molmoact \
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
  echo "=== molmoact <- $(basename "$record") ==="
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
    --output "$OUTPUT_DIR/molmoact_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  MolmoAct evaluation complete."
echo "  Reports:  ${OUTPUT_DIR}/replay_*.json"
echo "  Summary:  ${OUTPUT_DIR}/molmoact_summary.json"
echo "========================================"
