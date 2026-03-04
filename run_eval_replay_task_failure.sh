#!/usr/bin/env bash
# Replay pre-recorded task_failure attack prompts on victim VLA models.
#
# Uses perturbed instructions from the openpi_pi0 and openpi_pi05 recording
# sessions and evaluates each VLA on both original and perturbed instructions
# — WITHOUT running the attack agent.  Only 1 GPU needed.
#
# Environment strategy:
#   - runpod env:         Main process (LIBERO env, orchestration)
#   - vla_models env:     OpenVLA, LightVLA, ECoT, DeepThinkVLA (subprocess)
#   - vla_molmoact env:   MolmoAct (subprocess, needs transformers >= 4.45)
#   - vla_internvla env:  InternVLA-M1 (subprocess, needs transformers 4.52)
#
# Usage:
#   bash run_eval_replay_task_failure.sh                            # all models, both sources, GPU 0
#   bash run_eval_replay_task_failure.sh openvla                    # single model
#   bash run_eval_replay_task_failure.sh openvla lightvla           # two models sequentially
#   bash run_eval_replay_task_failure.sh --gpu 2 openvla            # use GPU 2
#   bash run_eval_replay_task_failure.sh --source pi05 openvla      # only pi05 source
#   bash run_eval_replay_task_failure.sh --source pi0 --gpu 1 ecot  # pi0 source on GPU 1
#   bash run_eval_replay_task_failure.sh --no-aggregate openvla     # skip aggregation step
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
OUTPUT_DIR="outputs/replay_eval_task_failure"
VLA_GPU=0
SOURCE_FILTER=""   # empty = both sources
DO_AGGREGATE=true

RECORD_DIR="outputs/agent_output_records_task_failure_2"

# ---- Parse flags (before positional model args) ----
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      VLA_GPU="$2"; shift 2 ;;
    --source)
      SOURCE_FILTER="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --record-dir)
      RECORD_DIR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --no-aggregate)
      DO_AGGREGATE=false; shift ;;
    -*)
      echo "Unknown flag: $1"; exit 1 ;;
    *)
      MODELS+=("$1"); shift ;;
  esac
done

# Default model list if none specified
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(
    openvla
    lightvla
    deepthinkvla
    ecot
    molmoact
    internvla_m1
  )
fi

# Build source record list based on --source filter
ATTACK_RECORDS=()
case "${SOURCE_FILTER,,}" in
  pi0)
    ATTACK_RECORDS=("${RECORD_DIR}/task_failure_openpi_pi0.json") ;;
  pi05|pi0.5)
    ATTACK_RECORDS=("${RECORD_DIR}/task_failure_openpi_pi05.json") ;;
  ""|both|all)
    ATTACK_RECORDS=(
      "${RECORD_DIR}/task_failure_openpi_pi0.json"
      "${RECORD_DIR}/task_failure_openpi_pi05.json"
    ) ;;
  *)
    echo "ERROR: Unknown source '${SOURCE_FILTER}'. Use: pi0, pi05, or both (default)."
    exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"

run_one() {
  local model=$1
  local record=$2
  local source_name
  source_name=$(basename "$record" .json | sed 's/task_failure_//')

  local log_file="${OUTPUT_DIR}/${model}_from_${source_name}.log"

  echo "[GPU ${VLA_GPU}] Replaying attacks on ${model} (source: ${source_name})"

  python eval_replay_attack.py \
    --victim "$model" \
    --vla_gpu "$VLA_GPU" \
    --attack_record "$record" \
    --seed "$SEED" \
    --replan_steps "$REPLAN_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$log_file"

  echo "[GPU ${VLA_GPU}] Done: ${model} (source: ${source_name})"
}

echo "========================================"
echo "  Replay Attack Evaluation (task_failure)"
echo "  Models:  ${MODELS[*]}"
echo "  Sources: ${ATTACK_RECORDS[*]}"
echo "  GPU:     ${VLA_GPU}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "========================================"

for model in "${MODELS[@]}"; do
  for record in "${ATTACK_RECORDS[@]}"; do
    if [[ ! -f "$record" ]]; then
      echo "WARNING: Attack record not found: ${record}"
      continue
    fi
    echo ""
    echo "=== ${model} ← $(basename "$record") ==="
    run_one "$model" "$record"
  done
done

# ---- Aggregate results ----
if [[ "$DO_AGGREGATE" == true ]]; then
  echo ""
  echo "========================================"
  echo "  Running cross-model aggregation..."
  echo "========================================"

  python aggregate_replay_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/cross_model_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log"
fi

echo ""
echo "========================================"
echo "  All replay evaluations complete."
echo "  Per-model reports: ${OUTPUT_DIR}/replay_*.json"
echo "  Cross-model summary: ${OUTPUT_DIR}/cross_model_summary.json"
echo "========================================"
