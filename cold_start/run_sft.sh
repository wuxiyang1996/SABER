#!/usr/bin/env bash
# Cold-start SFT: fine-tune Qwen2.5-3B-Instruct on GPT-5 Mini attack trajectories.
#
# Full-parameter SFT (no LoRA). Qwen2.5-3B is ~6GB in bf16, fits easily
# on a single A100. Gives a stronger cold-start than LoRA.
#
# After SFT completes, start GRPO with:
#   python train_vla.py --resume --base_model outputs/sft_runs/<run>/model
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- Conda activation ----
for p in /workspace/miniforge3 /opt/miniforge3 /opt/miniconda3 ~/miniforge3 ~/miniconda3; do
  if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
    source "${p}/etc/profile.d/conda.sh"
    break
  fi
done
conda activate runpod 2>/dev/null || conda activate vast 2>/dev/null || true

if [[ -x /venv/vast/bin/python ]]; then
  PYTHON=/venv/vast/bin/python
elif command -v python &>/dev/null; then
  PYTHON=python
else
  PYTHON=python3
fi

# ---- Objective tag and run selector ----
# objective_tag: task_failure | action_inflation | thinking_inflation | hallucination | constraint_violation
# run_selector: which run when multiple exist — number (1=latest, 2=second) or timestamp substring (e.g. 083338)
OBJECTIVE_TAG="${OBJECTIVE_TAG:-task_failure}"
RUN_SELECTOR="${RUN_TAG:-}"   # env RUN_TAG can set default run (e.g. 083338)
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  if [[ -d "${1:-}" ]]; then
    DATA_DIR="$1"
    shift
  else
    OBJECTIVE_TAG="$1"
    shift
    if [[ -n "${1:-}" && "${1:-}" != --* && ! -d "${1:-}" ]]; then
      RUN_SELECTOR="$1"
      shift
    fi
  fi
fi

# ---- Resolve data directory ----
if [[ -z "${DATA_DIR:-}" ]]; then
  # List all matching runs (newest first)
  MATCHING_DIRS=()
  for d in $(ls -dt cold_start/outputs/cold_start__*__${OBJECTIVE_TAG}__* 2>/dev/null); do
    [[ -d "$d" ]] && MATCHING_DIRS+=( "$d" )
  done
  if [[ ${#MATCHING_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No cold-start data directory found for objective tag: $OBJECTIVE_TAG"
    echo "Usage: $0 [objective_tag] [run_selector] [extra args...]"
    echo "  objective_tag: task_failure | action_inflation | ... (default: task_failure)"
    echo "  run_selector:  when multiple runs exist — 1=latest, 2=second latest, or timestamp e.g. 083338"
    echo "  Or pass full path: $0 /path/to/cold_start/outputs/..."
    exit 1
  fi
  if [[ -n "$RUN_SELECTOR" ]]; then
    if [[ "$RUN_SELECTOR" =~ ^[0-9]+$ ]]; then
      # Nth run (1 = newest)
      IDX=$((RUN_SELECTOR - 1))
      if [[ $IDX -lt 0 || $IDX -ge ${#MATCHING_DIRS[@]} ]]; then
        echo "ERROR: Run index $RUN_SELECTOR out of range (1..${#MATCHING_DIRS[@]}). Available:"
        for i in "${!MATCHING_DIRS[@]}"; do echo "  $((i+1)): ${MATCHING_DIRS[$i]}"; done
        exit 1
      fi
      DATA_DIR="${MATCHING_DIRS[$IDX]}"
    else
      # Substring match (e.g. timestamp 083338)
      FOUND=""
      for d in "${MATCHING_DIRS[@]}"; do
        if [[ "$d" == *"$RUN_SELECTOR"* ]]; then FOUND="$d"; break; fi
      done
      if [[ -z "$FOUND" ]]; then
        echo "ERROR: No run matching '$RUN_SELECTOR'. Available:"
        for i in "${!MATCHING_DIRS[@]}"; do echo "  $((i+1)): ${MATCHING_DIRS[$i]}"; done
        exit 1
      fi
      DATA_DIR="$FOUND"
    fi
  else
    DATA_DIR="${MATCHING_DIRS[0]}"
    if [[ ${#MATCHING_DIRS[@]} -gt 1 ]]; then
      echo "Multiple runs for $OBJECTIVE_TAG (using latest). Specify with: $0 $OBJECTIVE_TAG 2  or  $0 $OBJECTIVE_TAG <timestamp>"
      for i in "${!MATCHING_DIRS[@]}"; do echo "  $((i+1)): ${MATCHING_DIRS[$i]}"; done
    fi
  fi
fi
if [[ -z "$DATA_DIR" || ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Data directory not found: ${DATA_DIR:-<empty>}"
  exit 1
fi

# ---- Multi-GPU: use torchrun for data-parallel SFT (set N_GPUS=1 to avoid OOM) ----
N_GPUS="${N_GPUS:-1}"
if [[ "$N_GPUS" -gt 1 ]]; then
  SFT_LAUNCH=("$PYTHON" -m torch.distributed.run "--nproc_per_node=$N_GPUS" cold_start/sft_train.py)
else
  SFT_LAUNCH=("$PYTHON" -m cold_start.sft_train)
fi

echo "============================================================"
echo "Cold-Start SFT Training (LoRA)"
echo "============================================================"
echo "Objective: $OBJECTIVE_TAG"
echo "Python:     $PYTHON"
echo "Data dir:   $DATA_DIR"
echo "GPUs:       $N_GPUS"
echo "============================================================"

"${SFT_LAUNCH[@]}" \
  --data_dir "$DATA_DIR" \
  --output_dir outputs/sft_runs \
  --base_model Qwen/Qwen2.5-3B-Instruct \
  --max_seq_length 4096 \
  --num_epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --min_reward 0.3 \
  "$@"
