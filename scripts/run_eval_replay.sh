#!/usr/bin/env bash
# Unified replay evaluation: replay pre-recorded attack prompts on any victim VLA.
#
# Uses eval_replay_attack.py which executes each VLA twice per episode:
#   1. Baseline rollout (original instruction)
#   2. Attack rollout (perturbed instruction)
#
# Only 1 GPU needed (no attack agent inference).
#
# Usage:
#   bash scripts/run_eval_replay.sh --victim openvla --record outputs/.../task_failure_openpi_pi05.json
#   bash scripts/run_eval_replay.sh --victim internvla_m1 --record outputs/.../constraint_violation_openpi_pi05.json
#   bash scripts/run_eval_replay.sh --all-victims --record outputs/.../task_failure_openpi_pi05.json
#
# Options:
#   --victim MODEL       Target VLA model name
#   --all-victims        Run all standard victim models sequentially
#   --record PATH        Path to attack record JSON
#   --gpu N              GPU to use (default: 0, model-specific override)
#   --replan N           Replan steps (default: model-specific)
#   --max-steps N        Max rollout steps (default: model-specific or 400)
#   --episodes N         Episodes per task (default: use all from record)
#   --output-dir DIR     Output directory (default: outputs/eval_result)
#   --seed N             Random seed (default: 42)
#   --no-aggregate       Skip aggregation step
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- Parse arguments ----
VICTIM=""
ALL_VICTIMS=false
ATTACK_RECORD=""
VLA_GPU=""
REPLAN_STEPS=""
MAX_STEPS=""
EPISODES_PER_TASK=""
OUTPUT_DIR="outputs/eval_result"
SEED=42
NUM_WORKERS=1
DO_AGGREGATE=true
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --victim)         VICTIM="$2"; shift 2 ;;
    --all-victims)    ALL_VICTIMS=true; shift ;;
    --record)         ATTACK_RECORD="$2"; shift 2 ;;
    --gpu)            VLA_GPU="$2"; shift 2 ;;
    --replan)         REPLAN_STEPS="$2"; shift 2 ;;
    --max-steps)      MAX_STEPS="$2"; shift 2 ;;
    --episodes)       EPISODES_PER_TASK="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --workers)        NUM_WORKERS="$2"; shift 2 ;;
    --no-aggregate)   DO_AGGREGATE=false; shift ;;
    -*)               echo "Unknown flag: $1"; exit 1 ;;
    *)                shift ;;
  esac
done

if [[ -z "$ATTACK_RECORD" ]]; then
  echo "ERROR: --record is required."
  echo "Usage: $0 --victim MODEL --record PATH [options]"
  exit 1
fi

if [[ "$ALL_VICTIMS" == false && -z "$VICTIM" ]]; then
  echo "ERROR: Either --victim MODEL or --all-victims is required."
  exit 1
fi

if [[ ! -f "$ATTACK_RECORD" ]]; then
  echo "ERROR: Attack record not found: ${ATTACK_RECORD}"
  exit 1
fi

# ---- Conda ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done
conda activate runpod

# ---- Env ----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1

# NOTE: Per-model conda environments are handled automatically by Python code.
# model_factory.py launches non-JAX VLA models in subprocesses using
# SubprocessVLAWrapper, which picks the correct conda env's Python binary
# via _MODEL_ENV_MAP. See installation/setup_vla_envs.sh for env creation.

# ---- Model-specific defaults ----
# Returns: default GPU, default replan_steps, default max_steps
get_model_defaults() {
  local model=$1
  case "$model" in
    internvla_m1)
      echo "0 5 400" ;;
    *)
      echo "0 5 400" ;;
  esac
}

setup_model_env() {
  local model=$1
  case "$model" in
    internvla_m1)
      export TORCH_HOME=/workspace/.cache_torch
      mkdir -p "$TORCH_HOME"
      if [[ ! -L /root/.cache/torch/hub ]]; then
        mkdir -p /root/.cache/torch
        ln -sfn /workspace/.cache_torch_hub /root/.cache/torch/hub 2>/dev/null || true
      fi
      ;;
  esac
}

# ---- Single-model eval function ----
run_victim() {
  local model=$1

  local defaults
  defaults=$(get_model_defaults "$model")
  local default_gpu default_replan default_max_steps
  default_gpu=$(echo "$defaults" | cut -d' ' -f1)
  default_replan=$(echo "$defaults" | cut -d' ' -f2)
  default_max_steps=$(echo "$defaults" | cut -d' ' -f3)

  local gpu="${VLA_GPU:-$default_gpu}"
  local replan="${REPLAN_STEPS:-$default_replan}"
  local max_steps="${MAX_STEPS:-$default_max_steps}"

  setup_model_env "$model"

  local source_name
  source_name=$(basename "$ATTACK_RECORD" .json)
  local log_file="${OUTPUT_DIR}/${model}_from_${source_name}.log"

  echo "========================================"
  echo "  Replay Attack Evaluation"
  echo "  Victim:     ${model}"
  echo "  Source:     ${ATTACK_RECORD}"
  echo "  GPU:        ${gpu}"
  echo "  Seed:       ${SEED}"
  echo "  Replan:     ${replan}"
  echo "  Max steps:  ${max_steps}"
  echo "  Workers:    ${NUM_WORKERS}"
  echo "  Output:     ${OUTPUT_DIR}/"
  echo "========================================"
  echo ""

  local cmd_args=(
    --victim "$model"
    --vla_gpu "$gpu"
    --attack_record "$ATTACK_RECORD"
    --seed "$SEED"
    --replan_steps "$replan"
    --output_dir "$OUTPUT_DIR"
  )

  [[ "$max_steps" != "400" ]] && cmd_args+=(--max_steps "$max_steps")
  [[ -n "$EPISODES_PER_TASK" ]] && cmd_args+=(--episodes_per_task "$EPISODES_PER_TASK")

  if [[ "$NUM_WORKERS" -le 1 ]]; then
    python eval_replay_attack.py "${cmd_args[@]}" 2>&1 | tee "$log_file"
  else
    local pids=()
    for ((w=0; w<NUM_WORKERS; w++)); do
      local w_log="${OUTPUT_DIR}/${model}_from_${source_name}_w${w}.log"
      echo "[Worker $w/${NUM_WORKERS}] Starting on GPU ${gpu} ..."
      python eval_replay_attack.py "${cmd_args[@]}" \
        --num_workers "$NUM_WORKERS" --worker_id "$w" \
        > "$w_log" 2>&1 &
      pids+=($!)
    done
    echo "Launched ${NUM_WORKERS} workers: PIDs ${pids[*]}"
    local fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || ((fail++))
    done
    echo "All workers finished (${fail} failures)."
    python -c "
import json, glob, os, sys
pattern = os.path.join('$OUTPUT_DIR', 'replay_*_${model}_from_*_w*.json')
parts = sorted(glob.glob(pattern))
if not parts:
    print('No worker result files found to merge.'); sys.exit(0)
merged = []
for p in parts:
    with open(p) as f: merged.extend(json.load(f)['per_episode'])
    os.remove(p)
out = parts[0].rsplit('_w', 1)[0] + '.json'
with open(out) as f: base = json.load(f)
base['per_episode'] = merged
base['config']['num_workers'] = $NUM_WORKERS
with open(out, 'w') as f: json.dump(base, f, indent=2, default=str)
print(f'Merged {len(merged)} episodes into {out}')
" 2>&1 || echo "(Merge step skipped)"
  fi

  echo ""
  echo "[GPU ${gpu}] Replay evaluation done: ${model} <- ${source_name}"

  if [[ "$DO_AGGREGATE" == true ]]; then
    echo ""
    python aggregate_replay_results.py \
      --input_dir "$OUTPUT_DIR" \
      --output "$OUTPUT_DIR/${model}_eval_summary.json" \
      2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
  fi
}

# ---- Main ----
mkdir -p "$OUTPUT_DIR"

if [[ "$ALL_VICTIMS" == true ]]; then
  VICTIM_LIST=(openpi_pi05 openvla ecot deepthinkvla molmoact internvla_m1)
  for v in "${VICTIM_LIST[@]}"; do
    echo ""
    echo "############### ${v} ###############"
    run_victim "$v"
  done
else
  run_victim "$VICTIM"
fi

echo ""
echo "========================================"
echo "  All replay evaluations complete."
echo "  Reports: ${OUTPUT_DIR}/replay_*.json"
echo "========================================"
