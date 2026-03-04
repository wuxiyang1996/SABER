#!/usr/bin/env bash
# Evaluate AgiBot-World GO-1 on LIBERO using pre-recorded attack prompts from openpi_pi05.
#
# This script can do BOTH in one run:
#   (1) Start the GO-1 server (if GO1_MODEL_PATH and GO1_DATA_STATS_PATH are set)
#   (2) Run the replay evaluation (client)
#
# Mode A — Start server and run eval (single script):
#   export GO1_MODEL_PATH=/path/to/checkpoint
#   export GO1_DATA_STATS_PATH=/path/to/dataset_stats.json
#   export AGIBOT_WORLD_ROOT=/path/to/AgiBot-World   # optional; default below
#   bash run_eval_go1_agibot_from_pi05.sh
#
# Mode B — Server already running elsewhere:
#   export GO1_SERVER=127.0.0.1:9000   # optional
#   bash run_eval_go1_agibot_from_pi05.sh
#
# First-time setup: bash scripts/setup_go1_agibot.sh [--download] [--no-flash-attn]
# Optional flags: --output-dir, --seed, --replan, --no-aggregate
# See: https://github.com/OpenDriveLab/AgiBot-World/blob/main/evaluate/libero/README.md
set -euo pipefail

cd "$(dirname "$0")"

# ---- Conda ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  [[ -f "${p}/etc/profile.d/conda.sh" ]] && { source "${p}/etc/profile.d/conda.sh"; break; }
done

# ---- Defaults ----
SEED=42
REPLAN_STEPS=5
DO_AGGREGATE=true
GO1_PORT="${GO1_PORT:-9000}"
GO1_SERVER="${GO1_SERVER:-127.0.0.1:${GO1_PORT}}"

# AgiBot-World repo and GO-1 deploy paths (override with env vars if needed)
AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-$(pwd)/repos/AgiBot-World}"
GO1_MODEL_PATH="${GO1_MODEL_PATH:-$(pwd)/repos/AgiBot-World/checkpoints/GO-1}"
GO1_DATA_STATS_PATH="${GO1_DATA_STATS_PATH:-$(pwd)/repos/AgiBot-World/checkpoints/GO-1/dataset_stats.json}"
export AGIBOT_WORLD_ROOT GO1_MODEL_PATH GO1_DATA_STATS_PATH GO1_PORT GO1_SERVER

VICTIM="go1"
ATTACK_RECORD="outputs/agent_output_records_task_failure_2/task_failure_openpi_pi05.json"
OUTPUT_DIR="outputs/eval_result"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case "$1" in
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

# ---- Optional: start GO-1 server in background ----
GO1_SERVER_PID=""
if [[ -n "${GO1_MODEL_PATH:-}" && -n "${GO1_DATA_STATS_PATH:-}" ]]; then
  if [[ ! -d "$AGIBOT_WORLD_ROOT" ]]; then
    echo "ERROR: AGIBOT_WORLD_ROOT not found: ${AGIBOT_WORLD_ROOT}"
    echo "Set AGIBOT_WORLD_ROOT to the AgiBot-World repo clone, or start the server manually."
    exit 1
  fi
  if [[ ! -f "$GO1_MODEL_PATH" && ! -d "$GO1_MODEL_PATH" ]]; then
    echo "ERROR: GO1_MODEL_PATH not found: ${GO1_MODEL_PATH}"
    exit 1
  fi
  if [[ ! -f "$GO1_DATA_STATS_PATH" ]]; then
    echo "ERROR: GO1_DATA_STATS_PATH not found: ${GO1_DATA_STATS_PATH}"
    exit 1
  fi

  echo "========================================"
  echo "  Starting GO-1 server (background)"
  echo "  Model:   ${GO1_MODEL_PATH}"
  echo "  Stats:   ${GO1_DATA_STATS_PATH}"
  echo "  Port:    ${GO1_PORT}"
  echo "========================================"

  (
    for _p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
      [[ -f "${_p}/etc/profile.d/conda.sh" ]] && { source "${_p}/etc/profile.d/conda.sh"; break; }
    done
    conda activate go1 2>/dev/null || conda activate base
    cd "$AGIBOT_WORLD_ROOT"
    exec python evaluate/deploy.py \
      --model_path "$GO1_MODEL_PATH" \
      --data_stats_path "$GO1_DATA_STATS_PATH" \
      --host 0.0.0.0 \
      --port "$GO1_PORT"
  ) > "${OUTPUT_DIR}/go1_server.log" 2>&1 &
  GO1_SERVER_PID=$!

  # Wait for server to be ready (POST /act returns some HTTP code)
  echo "Waiting for GO-1 server on port ${GO1_PORT}..."
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "http://127.0.0.1:${GO1_PORT}/act" 2>/dev/null || echo "000")
    if [[ -n "$code" && "$code" != "000" ]]; then
      echo "GO-1 server ready (HTTP $code)."
      break
    fi
    if ! kill -0 "$GO1_SERVER_PID" 2>/dev/null; then
      echo "ERROR: GO-1 server process exited. Check ${OUTPUT_DIR}/go1_server.log"
      exit 1
    fi
    sleep 2
  done
  if [[ "$code" == "000" || -z "$code" ]]; then
    echo "ERROR: GO-1 server did not become ready in time. Check ${OUTPUT_DIR}/go1_server.log"
    kill "$GO1_SERVER_PID" 2>/dev/null || true
    exit 1
  fi

  # Kill server on script exit
  cleanup_go1() {
    if [[ -n "$GO1_SERVER_PID" ]] && kill -0 "$GO1_SERVER_PID" 2>/dev/null; then
      echo "Stopping GO-1 server (PID $GO1_SERVER_PID)..."
      kill "$GO1_SERVER_PID" 2>/dev/null || true
      wait "$GO1_SERVER_PID" 2>/dev/null || true
    fi
  }
  trap cleanup_go1 EXIT
else
  # Mode B: server not started here — check it is reachable before running eval
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 -X POST -H "Content-Type: application/json" -d '{}' "http://127.0.0.1:${GO1_PORT}/act" 2>/dev/null || echo "000")
  code="${code//[$'\r\n']}"
  # 000 / 000000 / empty = connection refused or failed
  if [[ -z "$code" || "$code" =~ ^0+$ ]]; then
    echo "ERROR: GO-1 server is not reachable at 127.0.0.1:${GO1_PORT}"
    _DEFAULT_MODEL="$(pwd)/repos/AgiBot-World/checkpoints/GO-1"
    if [[ ! -d "$_DEFAULT_MODEL" ]]; then
      echo "  No checkpoint at ${_DEFAULT_MODEL}"
      echo "  Run once to download the model and enable auto-start:"
      echo "    bash scripts/setup_go1_agibot.sh --download"
      echo "  (Put dataset_stats.json in that checkpoint dir, or set GO1_DATA_STATS_PATH to your stats file.)"
    fi
    echo "  Or start the server manually in another terminal:"
    echo "    conda activate go1"
    echo "    cd ${AGIBOT_WORLD_ROOT}"
    echo "    python evaluate/deploy.py --model_path <path> --data_stats_path <path> --port ${GO1_PORT}"
    echo "  Or set GO1_MODEL_PATH and GO1_DATA_STATS_PATH and re-run this script to start the server automatically."
    exit 1
  fi
fi

# ---- Env for eval (client) ----
conda activate runpod
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUTF8=1
export GO1_SERVER="127.0.0.1:${GO1_PORT}"

echo ""
echo "========================================"
echo "  AgiBot GO-1 Replay Attack Evaluation"
echo "  Victim:  ${VICTIM} (AgiBot-World server)"
echo "  Server:  ${GO1_SERVER}"
echo "  Source:  ${ATTACK_RECORD}"
echo "  Seed:    ${SEED}"
echo "  Replan:  ${REPLAN_STEPS}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "========================================"
echo ""

python eval_replay_attack.py \
  --victim "$VICTIM" \
  --vla_gpu "0" \
  --attack_record "$ATTACK_RECORD" \
  --seed "$SEED" \
  --replan_steps "$REPLAN_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Replay evaluation done: ${VICTIM} <- ${SOURCE_NAME}"

# ---- Aggregate results ----
if [[ "$DO_AGGREGATE" == true ]] && command -v python &>/dev/null; then
  echo ""
  echo "========================================"
  echo "  Aggregating results..."
  echo "========================================"

  python aggregate_replay_results.py \
    --input_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/go1_eval_summary.json" \
    2>&1 | tee "${OUTPUT_DIR}/aggregation.log" || true
fi

echo ""
echo "========================================"
echo "  AgiBot GO-1 evaluation complete."
echo "  Report:   ${OUTPUT_DIR}/replay_task_failure_${VICTIM}_from_openpi_pi05.json"
echo "  Summary:  ${OUTPUT_DIR}/go1_eval_summary.json"
echo "========================================"
