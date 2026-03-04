#!/usr/bin/env bash
# Evaluate InspireVLA (minivla-inspire-libero-union4) on LIBERO using pre-recorded attack prompts from openpi_pi05.
#
# Reads original_instruction / perturbed_instruction from the openpi_pi05 attack
# record, cross-checks original_instruction against LIBERO ground truth, and runs
# InspireVLA twice per episode (baseline + attack).
#
# Output goes to outputs/eval_result/ in the same JSON format as the recorded replay.
#
# InspireVLA specifics:
#   - Model: InspireVLA/minivla-inspire-libero-union4 (Hugging Face)
#   - Single checkpoint for all LIBERO suites (union4)
#   - Runs in vla_models conda env (subprocess isolation)
#
# Usage:
#   bash run_eval_inspirevla_from_pi05.sh                 # GPU 0
#   bash run_eval_inspirevla_from_pi05.sh --gpu 2        # GPU 2
#   bash run_eval_inspirevla_from_pi05.sh --no-aggregate # skip aggregation
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
NUM_WORKERS=1
FAST_FLAG=""

VICTIM="inspirevla"
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
    --workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --fast)
      FAST_FLAG="--fast"; shift ;;
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
echo "  InspireVLA Replay Attack Evaluation"
echo "  Victim:  ${VICTIM} (InspireVLA/minivla-inspire-libero-union4)"
echo "  Source:  ${ATTACK_RECORD}"
echo "  GPU:     ${VLA_GPU}"
echo "  Seed:    ${SEED}"
echo "  Replan:  ${REPLAN_STEPS}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Fast:    ${FAST_FLAG:-off}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "========================================"
echo ""

if [[ "$NUM_WORKERS" -le 1 ]]; then
  python eval_replay_attack.py \
    --victim "$VICTIM" \
    --vla_gpu "$VLA_GPU" \
    --attack_record "$ATTACK_RECORD" \
    --seed "$SEED" \
    --replan_steps "$REPLAN_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    $FAST_FLAG \
    2>&1 | tee "$LOG_FILE"
else
  PIDS=()
  for ((w=0; w<NUM_WORKERS; w++)); do
    W_LOG="${OUTPUT_DIR}/${VICTIM}_from_${SOURCE_NAME}_w${w}.log"
    echo "[Worker $w/${NUM_WORKERS}] Starting on GPU ${VLA_GPU} ..."
    python eval_replay_attack.py \
      --victim "$VICTIM" \
      --vla_gpu "$VLA_GPU" \
      --attack_record "$ATTACK_RECORD" \
      --seed "$SEED" \
      --replan_steps "$REPLAN_STEPS" \
      --output_dir "$OUTPUT_DIR" \
      --num_workers "$NUM_WORKERS" \
      --worker_id "$w" \
      $FAST_FLAG \
      > "$W_LOG" 2>&1 &
    PIDS+=($!)
  done
  echo "Launched ${NUM_WORKERS} workers: PIDs ${PIDS[*]}"
  echo "Waiting for all workers ..."
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAIL++))
  done
  echo "All workers finished (${FAIL} failures)."
  # Merge per-worker results
  python -c "
import json, glob, os, sys
pattern = os.path.join('$OUTPUT_DIR', 'replay_*_${VICTIM}_from_*_w*.json')
parts = sorted(glob.glob(pattern))
if not parts:
    print('No worker result files found to merge.'); sys.exit(0)
merged_eps = []
for p in parts:
    with open(p) as f:
        merged_eps.extend(json.load(f)['per_episode'])
    os.remove(p)
with open(parts[0].rsplit('_w', 1)[0] + '.json') as f:
    base = json.load(f)
base['per_episode'] = merged_eps
base['config']['num_workers'] = $NUM_WORKERS
out = parts[0].rsplit('_w', 1)[0] + '.json'
with open(out, 'w') as f:
    json.dump(base, f, indent=2, default=str)
print(f'Merged {len(merged_eps)} episodes into {out}')
" 2>&1 || echo "(Merge step skipped — run manually if needed)"
fi

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
    --output "$OUTPUT_DIR/inspirevla_eval_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  InspireVLA evaluation complete."
echo "  Report:   ${OUTPUT_DIR}/replay_task_failure_${VICTIM}_from_openpi_pi05.json"
echo "  Summary:  ${OUTPUT_DIR}/inspirevla_eval_summary.json"
echo "========================================"
