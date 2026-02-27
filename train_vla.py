"""GRPO training for the VLA adversarial attack agent.

Trains an LLM attack agent to perturb instructions/observations fed to
π0.5 in LIBERO, optimising for a single declared attack objective via
Group Relative Policy Optimisation (GRPO).

Train/test split (7/3 per suite, 4 eval suites):
  - **Train**: tasks 0-6 from each suite = 28 tasks × 1 init state = 28 scenarios
  - **Test** : tasks 7-9 from each suite = 12 tasks × 10 init states = 120 episodes

Speedup defaults: replan_steps=5, episodes_per_task=1, num_epochs=1.

Usage
-----
    # Default: 7/3 split, train on 28 tasks, eval on 12 held-out tasks
    python train_vla.py

    # Custom suite(s)
    python train_vla.py --task_suite libero_spatial,libero_object

    # Override split
    python train_vla.py --task_ids 0-4 --eval_task_ids 5-9

    # Skip post-training evaluation
    python train_vla.py --skip_eval

The script:
  1. Loads the Pi0.5 VLA model (once, shared across all rollouts).
  2. Builds ``VLAAttackScenario`` objects for the declared tasks.
  3. Runs the GRPO training loop:
       - ``art.gather_trajectory_groups`` parallelises rollouts.
       - Each rollout calls ``vla_attack_rollout`` which:
           a) runs the clean baseline VLA episode,
           b) lets the attack agent perturb via tools,
           c) runs the perturbed VLA episode,
           d) computes reward.
       - The ``LocalBackend`` trains the model on trajectory groups.
  4. Runs post-training evaluation on the eval suite (unless ``--skip_eval``).

Reference:
  - ART GRPO: https://art.openpipe.ai
  - MCP-RL example: ART/examples/mcp-rl/
"""

from __future__ import annotations

import os
import sys
import time
import warnings

# Suppress JAX/Flax deprecation warnings from Pi0.5/openpi (ShapeDtypeStruct, wrap_init DebugInfo).
# These come from site-packages and will be fixed in future JAX/Flax releases.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="flax.core.scope",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="jax.extend.linear_util",
)

# ---- GPU layout: resolve early so CUDA_VISIBLE_DEVICES is set before
#      any CUDA / PyTorch / JAX import.
#
#   GPU 0,1,2 (default, or --vla_gpus) →  Pi0.5 VLA model(s) — rollouts (JAX)
#   GPU 3 (default, or --attack_gpus) →  Attack agent training (vLLM / ART)
#
# On SLURM, CUDA_VISIBLE_DEVICES is pre-set to *all* allocated GPUs
# (e.g. "0,1,2,3" or "4,5,6,7").  JAX initialises on every visible GPU,
# so we must restrict visibility to only the GPUs we need *before* any
# framework import.  --vla_gpu and --attack_gpus are *logical indices*
# into the SLURM-visible list, NOT raw physical IDs.

def _early_resolve_vla_gpus() -> list[int]:
    raw = None
    for i, tok in enumerate(sys.argv):
        if tok in ("--vla_gpu", "--vla_gpus") and i + 1 < len(sys.argv):
            raw = sys.argv[i + 1]
            break
        if tok.startswith("--vla_gpu=") or tok.startswith("--vla_gpus="):
            raw = tok.split("=", 1)[1]
            break
    if raw is None:
        raw = os.environ.get("VLA_GPUS", os.environ.get("VLA_GPU", "0,1,2"))
    return [int(g.strip()) for g in raw.split(",")]

_VLA_GPUS = _early_resolve_vla_gpus()

def _early_resolve_attack_gpus() -> str:
    for i, tok in enumerate(sys.argv):
        if tok == "--attack_gpus" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith("--attack_gpus="):
            return tok.split("=", 1)[1]
    return os.environ.get("ATTACK_GPUS", "")

_ATTACK_GPUS_RAW = _early_resolve_attack_gpus()

# Map logical indices → physical GPU IDs from the current visible set.
_orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
_orig_gpu_list = [g.strip() for g in _orig_visible.split(",") if g.strip()]

def _logical_to_physical(idx: int) -> str:
    """Map a logical GPU index to the physical ID from SLURM's list."""
    if _orig_gpu_list and idx < len(_orig_gpu_list):
        return _orig_gpu_list[idx]
    return str(idx)

# Default attack GPUs: all visible GPUs except the VLA GPU(s).
if _ATTACK_GPUS_RAW:
    _ATTACK_GPUS = _ATTACK_GPUS_RAW
else:
    _n_gpus = len(_orig_gpu_list) if _orig_gpu_list else max(2, max(_VLA_GPUS) + 2)
    _attack_indices = [i for i in range(_n_gpus) if i not in _VLA_GPUS]
    _ATTACK_GPUS = (
        ",".join(str(i) for i in _attack_indices)
        if _attack_indices
        else str(max(_VLA_GPUS) + 1)
    )

_needed_indices = sorted(set(_VLA_GPUS + [int(g) for g in _ATTACK_GPUS.split(",")]))

_ATTACK_GPUS_PHYSICAL = ",".join(
    _logical_to_physical(int(g)) for g in _ATTACK_GPUS.split(",")
)
# Only expose VLA GPUs to JAX during init — keeps JAX from claiming memory
# on the attack GPU(s).  CUDA_VISIBLE_DEVICES is switched to the attack
# GPU(s) later, right before vLLM starts.
_VLA_GPUS_PHYSICAL = ",".join(
    _logical_to_physical(i) for i in sorted(_VLA_GPUS)
)
os.environ["CUDA_VISIBLE_DEVICES"] = _VLA_GPUS_PHYSICAL

# JAX memory settings — don't pre-allocate; cap at 30% per GPU (~14 GiB on
# 48 GiB L40S).  Pi0.5 inference needs ~9 GiB; 0.30 leaves room for vLLM
# (attack agent) if sharing GPUs.  Override with env var if needed.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ---- Model / data cache directory --------------------------------------
# All caches (OpenPI checkpoints, HuggingFace, PyTorch, etc.) go under
# vlm-robot/.cache so shared clusters don't fill ~/.cache.
_CACHE_ROOT = os.path.realpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".cache"
))
os.environ.setdefault("OPENPI_DATA_HOME", _CACHE_ROOT)
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE_ROOT, "torch"))
try:
    os.makedirs(_CACHE_ROOT, exist_ok=True)
except OSError:
    pass

# ---- ART output (LoRA checkpoints, logs, trajectories) ------------------
# Store under agent_attack_framework/outputs so everything stays in the project.
_AGENT_ATTACK_ROOT = os.path.dirname(os.path.abspath(__file__))
_ART_OUTPUT_ROOT = os.path.realpath(os.path.join(_AGENT_ATTACK_ROOT, "outputs"))
try:
    os.makedirs(_ART_OUTPUT_ROOT, exist_ok=True)
except OSError:
    pass

# ---- Headless rendering (must be set before MuJoCo / PyOpenGL import) --
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import asyncio
import json
import logging

# ---- PyTorch 2.6+ compat: LIBERO init-state files contain numpy arrays.
# We patched libero/benchmark/__init__.py to use weights_only=False.
# (torch.load changed its default in PyTorch 2.6.)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

import art
import art.dev
from art.langgraph import wrap_rollout
from art.utils import iterate_dataset

from agent.vla_rollout import (
    ToolSet,
    VLAAttackScenario,
    build_scenarios,
    build_scenarios_multi_suite,
    clear_baseline_cache,
    parse_task_ids,
    set_vla_model,
    set_vla_models,
    vla_attack_rollout,
)
from rwd_func.rwd import AttackObjective

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OBJECTIVE = "action_inflation"
DEFAULT_TOOL_SETS = "token,char,prompt"
# Train/test split: 7/3 per suite across all 4 eval suites (Pi0.5 >95% success).
# Train: tasks 0-6 (28 tasks), Test: tasks 7-9 (12 tasks, 120 episodes).
DEFAULT_TASK_SUITE = "libero_spatial,libero_object,libero_goal,libero_10"
DEFAULT_TASK_IDS = "0-6"
DEFAULT_EVAL_TASK_IDS = "7-9"
DEFAULT_EVAL_EPISODES_PER_TASK = 5   # init states per test task (12 tasks × 5 = 60 episodes)
DEFAULT_EPISODES_PER_TASK = 1        # 1 init state per task; diversity comes from 28 tasks
DEFAULT_TRAJECTORIES_PER_GROUP = 8   # GRPO group size; matches 8 rollout workers per GPU
DEFAULT_GROUPS_PER_STEP = 4          # groups gathered before one train step
DEFAULT_NUM_EPOCHS = 3               # 3 passes over 28 scenarios = 21 training steps
DEFAULT_REPLAN_STEPS = 5             # matches official OpenPI LIBERO default
DEFAULT_SEED = 42                    # match eval.run_libero_eval default for comparable baseline success
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EVAL_STEPS = 5
DEFAULT_STEALTH_WEIGHT = 0.1
DEFAULT_MAX_EDIT_CHARS = 200  # hard budget: max Levenshtein char edits
# For action_inflation: use well above longest suite (520) so the attack rollout has
# headroom to succeed while taking more steps than baseline; otherwise timeouts zero the reward.
DEFAULT_MAX_STEPS = 800


# ============================================================================
# Main training loop
# ============================================================================

async def train(args: argparse.Namespace) -> None:
    """Run GRPO training for the VLA attack agent."""

    # --- Configure thread pool for parallel VLA rollouts ---------------
    # With async pipelining, the thread pool handles short-lived tasks:
    # VLA inference calls (GPU) and MuJoCo step chunks (CPU).  Multiple
    # MuJoCo chunks can run concurrently on separate CPU cores while one
    # VLA inference runs on GPU.  Use more threads than before.
    import concurrent.futures
    loop = asyncio.get_running_loop()
    n_workers = getattr(args, "rollout_workers", 4)
    pool_size = max(n_workers * 2, 8)
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=pool_size),
    )
    logger.info(
        "Thread pool: %d threads (%d rollout workers × 2) for pipelined "
        "VLA inference + MuJoCo stepping.", pool_size, n_workers,
    )

    # --- Parse objective and tool sets --------------------------------
    objective = AttackObjective(args.objective)
    tool_sets = [ToolSet(t.strip()) for t in args.tool_sets.split(",")]

    # Parse suite(s) — supports comma-separated multi-suite
    _suite_names = [s.strip() for s in args.task_suite.split(",")]
    _multi_suite = len(_suite_names) > 1
    suite_specs: list[tuple[str, list[int]]] = []
    for sn in _suite_names:
        sn_ids = parse_task_ids(args.task_ids, sn)
        suite_specs.append((sn, sn_ids))
    total_tasks = sum(len(ids) for _, ids in suite_specs)

    logger.info("=" * 60)
    logger.info("VLA Attack Agent — GRPO Training")
    logger.info("=" * 60)
    logger.info("  Objective:        %s", objective.value)
    logger.info("  Tool sets:        %s", [ts.value for ts in tool_sets])
    logger.info("  Max turns:        %d (ReAct tool-call rounds per episode)", args.max_turns)
    if _multi_suite:
        logger.info("  Task suites:      %s (%d tasks total)", _suite_names, total_tasks)
        for sn, sn_ids in suite_specs:
            logger.info("    %-20s %d tasks  IDs=%s", sn, len(sn_ids), sn_ids)
    else:
        logger.info("  Task suite:       %s", args.task_suite)
        logger.info("  Task IDs:         %s", suite_specs[0][1])
    logger.info("  Episodes/task:    %d", args.episodes_per_task)
    logger.info("  Trajs/group:      %d", args.trajectories_per_group)
    logger.info("  Groups/step:      %d", args.groups_per_step)
    logger.info("  Epochs:           %d", args.num_epochs)
    logger.info("  LR:               %s", args.learning_rate)
    logger.info("  Replan steps:     %d (VLA inference every N env steps)", args.replan_steps)
    logger.info("  Rollout workers:  %d threads", args.rollout_workers)
    logger.info("  Stealth weight:   %s", args.stealth_weight)
    logger.info("  No-attack penalty: %s (reward when no tool used)", args.no_attack_penalty)
    logger.info("  Short trajectory penalty: %s (when attack steps < %.0f%% of baseline)", args.short_trajectory_penalty, args.short_trajectory_ratio * 100)
    logger.info("  Max edit chars:   %d (hard budget)", args.max_edit_chars)
    logger.info("=" * 60)

    # --- Create timestamped run directory --------------------------------
    from datetime import datetime
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _agent_tag = args.model_name.replace("/", "_")
    _vla_tag = args.vla_config_name.replace("/", "_")
    _obj_tag = args.objective
    _all_tools = {"token", "char", "prompt"}
    _active_tools = {t.strip() for t in args.tool_sets.split(",")}
    _tools_tag = "all" if _active_tools >= _all_tools else "-".join(sorted(_active_tools))
    _run_name = f"{_agent_tag}__{_vla_tag}__{_timestamp}__{_obj_tag}__{_tools_tag}"
    _run_dir = os.path.join(_ART_OUTPUT_ROOT, "runs", _run_name)
    os.makedirs(_run_dir, exist_ok=True)
    logger.info("Run directory: %s", _run_dir)

    # File-based logging — mirror all log output to the run directory
    _file_handler = logging.FileHandler(os.path.join(_run_dir, "train.log"))
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
    ))
    logging.getLogger().addHandler(_file_handler)

    # Save full run config
    _run_config = {
        "timestamp": _timestamp,
        "run_name": _run_name,
        "vla_model": {
            "config_name": args.vla_config_name,
            "checkpoint": args.vla_checkpoint,
            "gpus": args.vla_gpus,
        },
        "attack_agent": {
            "model_name": args.model_name,
            "base_model": args.base_model,
            "project_name": args.project_name,
            "gpus": args.attack_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "training": {
            "objective": args.objective,
            "tool_sets": args.tool_sets,
            "task_suite": args.task_suite,
            "suite_specs": {sn: ids for sn, ids in suite_specs},
            "total_tasks": total_tasks,
            "episodes_per_task": args.episodes_per_task,
            "trajectories_per_group": args.trajectories_per_group,
            "groups_per_step": args.groups_per_step,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "stealth_weight": args.stealth_weight,
            "max_edit_chars": args.max_edit_chars,
            "max_turns": args.max_turns,
            "replan_steps": args.replan_steps,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "rollout_workers": args.rollout_workers,
        },
        "eval": {
            "eval_task_suite": args.eval_task_suite,
            "eval_task_ids": args.eval_task_ids,
            "eval_episodes_per_task": args.eval_episodes_per_task,
            "skip_eval": args.skip_eval,
        },
    }
    with open(os.path.join(_run_dir, "run_config.json"), "w") as _f:
        json.dump(_run_config, _f, indent=2, default=str)
    logger.info("Run config saved to %s/run_config.json", _run_dir)

    # Per-step metrics log (JSONL — one JSON object per training step)
    _metrics_log_path = os.path.join(_run_dir, "step_metrics.jsonl")

    # --- Resolve GPU layout from CLI -----------------------------------
    vla_gpus = [int(g.strip()) for g in args.vla_gpus.split(",")]
    attack_gpus = args.attack_gpus

    # CUDA_VISIBLE_DEVICES was set to VLA GPUs only, so JAX sees them
    # sequentially as devices 0..N-1.
    vla_gpus_sorted = sorted(vla_gpus)
    vla_jax_indices = [vla_gpus_sorted.index(g) for g in vla_gpus]

    logger.info(
        "GPU layout: VLA rollouts → GPU(s) %s (JAX idx %s)  |  Agent training → GPU(s) %s",
        vla_gpus, vla_jax_indices, attack_gpus,
    )
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s (VLA only, restricted from %s)",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), _orig_visible or "<unset>",
    )

    # --- Load Pi0.5 VLA model(s) on the VLA GPU(s) --------------------
    import jax
    try:
        jax_devices = jax.devices("gpu")
    except Exception as e:
        torch_info = ""
        try:
            import torch  # noqa: F401

            torch_info = (
                f"torch={torch.__version__}, "
                f"torch.cuda.is_available()={torch.cuda.is_available()}, "
                f"torch.cuda.device_count()={torch.cuda.device_count()}"
            )
        except Exception as _torch_e:  # pragma: no cover
            torch_info = f"torch import failed: {_torch_e}"

        raise RuntimeError(
            "JAX could not initialize a GPU backend (only CPU is available).\n"
            f"Details: {e}\n"
            f"CUDA visibility check: {torch_info}\n\n"
            "This VLA run requires NVIDIA GPUs for BOTH:\n"
            " - π0.5 (JAX) inference, and\n"
            " - the attack agent (vLLM via ART).\n\n"
            "Fix: run on a machine/session with GPUs exposed (e.g. request a GPU job, "
            "or ensure your container/runtime is started with NVIDIA device passthrough). "
            "Sanity checks:\n"
            " - `ls /dev/nvidia*` should show device nodes\n"
            " - `python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\"` "
            "should report True and >=1"
        ) from e
    logger.info(
        "JAX sees %d GPU(s): %s  (VLA will use jax device(s) %s)",
        len(jax_devices), jax_devices, vla_jax_indices,
    )
    for vla_gpu, vla_jax_idx in zip(vla_gpus, vla_jax_indices):
        if vla_jax_idx >= len(jax_devices):
            raise RuntimeError(
                f"--vla_gpus includes GPU {vla_gpu} mapped to JAX index {vla_jax_idx} "
                f"but JAX only sees {len(jax_devices)} GPU(s).  "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
            )

    # Lazy import to avoid loading heavy deps when just parsing args
    _libero_rollouts_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "libero_rollouts"),
    )
    if _libero_rollouts_dir not in sys.path:
        sys.path.insert(0, _libero_rollouts_dir)
    from pi05_libero_model import Pi05LiberoModel

    models_and_devices = []
    for i, (vla_gpu, vla_jax_idx) in enumerate(zip(vla_gpus, vla_jax_indices)):
        vla_jax_device = jax_devices[vla_jax_idx]
        logger.info(
            "Loading Pi0.5 VLA model %d/%d on JAX device %d (GPU %d) ...",
            i + 1, len(vla_gpus), vla_jax_idx, vla_gpu,
        )
        with jax.default_device(vla_jax_device):
            vla_model = Pi05LiberoModel(
                train_config_name=args.vla_config_name,
                checkpoint_path=args.vla_checkpoint,
                action_horizon=10,
                replan_steps=args.replan_steps,
            )
        models_and_devices.append((vla_model, vla_jax_device))

    set_vla_models(models_and_devices)
    logger.info(
        "Pi0.5 VLA model pool ready: %d instance(s) on GPU(s) %s.",
        len(vla_gpus), vla_gpus,
    )

    # --- JIT warmup: trigger XLA compilation on each GPU ---------------
    # First inference on each device compiles XLA kernels (30-60s).
    # Running a dummy inference here avoids that latency during training.
    import numpy as np
    logger.info("JIT warmup: running dummy inference on %d VLA instance(s)...", len(models_and_devices))
    _warmup_t0 = time.time()
    _dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    _dummy_state = np.zeros(8, dtype=np.float32)
    for i, (model, dev) in enumerate(models_and_devices):
        model.set_language("warmup")
        with jax.default_device(dev):
            _ = model.predict(_dummy_img, _dummy_img, _dummy_state)
        logger.info("  GPU %d warmup done.", vla_gpus[i])
    _warmup_elapsed = time.time() - _warmup_t0
    logger.info("JIT warmup complete in %.1fs. All kernels compiled.", _warmup_elapsed)

    # --- Switch CUDA_VISIBLE_DEVICES to attack GPUs for vLLM ----------
    # JAX init only saw VLA GPUs; now expose only the attack GPU(s) so
    # vLLM gets a clean GPU with no JAX memory footprint.
    os.environ["CUDA_VISIBLE_DEVICES"] = _ATTACK_GPUS_PHYSICAL
    logger.info(
        "Set CUDA_VISIBLE_DEVICES=%s for ART subprocess (vLLM).",
        _ATTACK_GPUS_PHYSICAL,
    )

    # --- Build scenarios ----------------------------------------------
    if _multi_suite:
        train_scenarios = build_scenarios_multi_suite(
            objective=objective,
            tool_sets=tool_sets,
            suite_specs=suite_specs,
            episodes_per_task=args.episodes_per_task,
            stealth_weight=args.stealth_weight,
            no_attack_penalty=args.no_attack_penalty,
            short_trajectory_penalty=args.short_trajectory_penalty,
            short_trajectory_ratio_threshold=args.short_trajectory_ratio,
            max_edit_chars=args.max_edit_chars,
            max_steps=args.max_steps,
            max_turns=args.max_turns,
            replan_steps=args.replan_steps,
            seed=args.seed,
        )
    else:
        train_scenarios = build_scenarios(
            objective=objective,
            tool_sets=tool_sets,
            task_suite_name=_suite_names[0],
            task_ids=suite_specs[0][1],
            episodes_per_task=args.episodes_per_task,
            stealth_weight=args.stealth_weight,
            no_attack_penalty=args.no_attack_penalty,
            short_trajectory_penalty=args.short_trajectory_penalty,
            short_trajectory_ratio_threshold=args.short_trajectory_ratio,
            max_edit_chars=args.max_edit_chars,
            max_steps=args.max_steps,
            max_turns=args.max_turns,
            replan_steps=args.replan_steps,
            seed=args.seed,
        )
    logger.info("Built %d training scenarios.", len(train_scenarios))

    # --- By default train from scratch; clear existing ART checkpoints ----
    # Unless --resume: remove this model's output dir so ART creates step 0
    # from the current base_model. Avoids LoRA tensor size mismatch when
    # base_model or model size changes.
    if not args.resume:
        try:
            from art.utils.output_dirs import get_output_dir_from_model_properties
            _model_dir = get_output_dir_from_model_properties(
                args.project_name, args.model_name, art_path=_ART_OUTPUT_ROOT,
            )
            if os.path.isdir(_model_dir):
                import shutil
                shutil.rmtree(_model_dir)
                logger.info("Training from scratch: removed existing model dir %s", _model_dir)
        except Exception as e:
            logger.warning("Could not clear model dir for from-scratch run: %s", e)
    else:
        logger.info("Resuming from existing checkpoints (--resume).")

    # --- ART model setup (vLLM on attack GPU(s)) ----------------------
    # Count how many GPUs were assigned to the attack agent
    n_attack_gpus = len(attack_gpus.split(","))
    logger.info(
        "Configuring attack model: %d GPU(s), mem_util=%.2f",
        n_attack_gpus, args.gpu_memory_utilization,
    )

    # gpu_memory_utilization goes into engine_args (the vLLM
    # AsyncEngineArgs), NOT init_args (unsloth model loader).
    # Putting it in init_args caused vLLM to silently use its own
    # default of 0.9 for gpu_memory_utilization.
    #
    # tensor_parallel_size is left at 1 (default).  So vLLM uses a single
    # GPU from the attack set; when --attack_gpus has multiple IDs, the
    # subprocess sees them (CUDA_VISIBLE_DEVICES) but vLLM uses the first.
    # Qwen2.5-3B (~6 GiB) fits on one L40S.  TP > 1 triggers vLLM
    # multiprocessing which hits a pydantic-core pickling bug.
    attack_model = art.TrainableModel(
        name=args.model_name,
        project=args.project_name,
        base_model=args.base_model,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=4096,
            ),
            engine_args=art.dev.EngineArgs(
                gpu_memory_utilization=args.gpu_memory_utilization,
            ),
        ),
    )

    # --- Patch model-service subprocess to capture stdout/stderr -----
    # The ART backend spawns a child process ("model-service") for
    # unsloth + vLLM.  If it crashes (e.g. SIGABRT), the real error is
    # lost unless we redirect output to a log file.
    _model_service_log = os.path.join(_ART_OUTPUT_ROOT, "model-service-debug.log")
    os.makedirs(os.path.dirname(_model_service_log), exist_ok=True)

    import mp_actors.move as _mp_move_module
    _orig_move_to_child = _mp_move_module.move_to_child_process

    def _move_to_child_with_log(obj, log_file=None, process_name=None):
        if process_name == "model-service" and log_file is None:
            log_file = _model_service_log
            logger.info("Model-service subprocess log → %s", log_file)
        return _orig_move_to_child(obj, log_file=log_file, process_name=process_name)

    _mp_move_module.move_to_child_process = _move_to_child_with_log

    from art.local.backend import LocalBackend
    import art.local.backend as _alb
    _alb.move_to_child_process = _move_to_child_with_log

    backend = LocalBackend(path=_ART_OUTPUT_ROOT)
    logger.info("ART output (checkpoints, logs): %s", _ART_OUTPUT_ROOT)
    try:
        await attack_model.register(backend)
    except RuntimeError as exc:
        if os.path.isfile(_model_service_log):
            with open(_model_service_log) as _f:
                _log_tail = _f.read()[-8000:]
            if _log_tail.strip():
                logger.error(
                    "model-service subprocess log (last 8 kB):\n%s", _log_tail,
                )
        raise

    # --- Training loop ------------------------------------------------
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=args.groups_per_step,
        num_epochs=args.num_epochs,
        initial_step=await attack_model.get_step(),
    )

    # wrap_rollout sets CURRENT_CONFIG so init_chat_model() inside vla_attack_rollout can run
    vla_rollout_wrapped = wrap_rollout(
        attack_model,
        lambda scenario: vla_attack_rollout(attack_model, scenario),
    )

    for batch in train_iterator:
        clear_baseline_cache()
        logger.info(
            "Step %d — gathering %d trajectory groups (rollout_type=attacked) ...",
            batch.step, len(batch.items),
        )

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    vla_rollout_wrapped(scenario)
                    for _ in range(args.trajectories_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train step {batch.step}",
        )

        logger.info(
            "Step %d — gathered %d groups, training ...",
            batch.step, len(groups),
        )

        result = await backend.train(
            attack_model, groups, learning_rate=args.learning_rate,
        )
        await attack_model.log(
            groups, metrics=result.metrics, step=result.step, split="train",
        )

        # Log summary statistics and evaluation metrics
        from rwd_func.metrics import compute_metrics, print_metrics
        step_metrics = compute_metrics(groups)
        print_metrics(step_metrics, step=batch.step, logger_fn=logger.info)

        # Append per-step metrics to the run's JSONL log
        _step_record = {
            "step": batch.step,
            "epoch": batch.epoch,
            "timestamp": datetime.now().isoformat(),
            "n_groups": len(groups),
            **step_metrics.to_dict(),
        }
        with open(_metrics_log_path, "a") as _mf:
            _mf.write(json.dumps(_step_record, default=str) + "\n")

        # Symlink latest checkpoint into the run directory
        _current_step = result.step
        try:
            from art.utils.output_dirs import (
                get_output_dir_from_model_properties,
                get_step_checkpoint_dir,
            )
            _model_out = get_output_dir_from_model_properties(
                args.project_name, args.model_name, art_path=_ART_OUTPUT_ROOT,
            )
            _ckpt_src = get_step_checkpoint_dir(_model_out, _current_step)
            if os.path.isdir(_ckpt_src):
                _ckpt_link = os.path.join(_run_dir, "checkpoints", f"{_current_step:04d}")
                os.makedirs(os.path.dirname(_ckpt_link), exist_ok=True)
                if not os.path.exists(_ckpt_link):
                    os.symlink(_ckpt_src, _ckpt_link)
        except Exception:
            pass

    logger.info("Training complete.")

    # ---- Post-training evaluation on held-out test tasks ------------------
    if args.skip_eval:
        logger.info("Skipping post-training evaluation (--skip_eval).")
    else:
        import time as _time
        from rwd_func.metrics import (
            compute_metrics_from_trajectories,
            print_metrics as _print_metrics,
            metrics_to_latex_row,
        )

        _eval_suite_str = args.eval_task_suite or args.task_suite
        _eval_suite_names = [s.strip() for s in _eval_suite_str.split(",")]
        _eval_multi = len(_eval_suite_names) > 1
        eval_suite_specs: list[tuple[str, list[int]]] = []
        for sn in _eval_suite_names:
            sn_ids = parse_task_ids(args.eval_task_ids, sn)
            eval_suite_specs.append((sn, sn_ids))
        eval_total = sum(len(ids) for _, ids in eval_suite_specs)

        logger.info("=" * 60)
        logger.info("Post-training evaluation (held-out test tasks)")
        logger.info("=" * 60)
        if _eval_multi:
            logger.info("  Eval suites:      %s (%d tasks, %d episodes total)",
                        _eval_suite_names, eval_total,
                        eval_total * args.eval_episodes_per_task)
            for sn, sn_ids in eval_suite_specs:
                logger.info("    %-20s %d tasks  IDs=%s", sn, len(sn_ids), sn_ids)
        else:
            logger.info("  Eval suite:       %s (%d tasks)", _eval_suite_names[0], eval_total)
        logger.info("  Episodes/task:    %d", args.eval_episodes_per_task)

        if _eval_multi:
            eval_scenarios = build_scenarios_multi_suite(
                objective=objective,
                tool_sets=tool_sets,
                suite_specs=eval_suite_specs,
                episodes_per_task=args.eval_episodes_per_task,
                stealth_weight=args.stealth_weight,
                no_attack_penalty=args.no_attack_penalty,
                short_trajectory_penalty=args.short_trajectory_penalty,
                short_trajectory_ratio_threshold=args.short_trajectory_ratio,
                max_edit_chars=args.max_edit_chars,
                max_steps=args.max_steps,
                max_turns=args.max_turns,
                replan_steps=args.replan_steps,
                seed=args.seed,
            )
        else:
            eval_scenarios = build_scenarios(
                objective=objective,
                tool_sets=tool_sets,
                task_suite_name=_eval_suite_names[0],
                task_ids=eval_suite_specs[0][1],
                episodes_per_task=args.eval_episodes_per_task,
                stealth_weight=args.stealth_weight,
                no_attack_penalty=args.no_attack_penalty,
                short_trajectory_penalty=args.short_trajectory_penalty,
                short_trajectory_ratio_threshold=args.short_trajectory_ratio,
                max_edit_chars=args.max_edit_chars,
                max_steps=args.max_steps,
                max_turns=args.max_turns,
                replan_steps=args.replan_steps,
                seed=args.seed,
            )
        logger.info("Built %d eval scenarios.", len(eval_scenarios))

        all_eval_trajectories = []
        t_eval_start = _time.time()

        for i, scenario in enumerate(eval_scenarios):
            clear_baseline_cache()
            logger.info(
                "Eval %d/%d: %s task %d, ep %d ...",
                i + 1, len(eval_scenarios),
                scenario.task_suite_name, scenario.task_id,
                scenario.episode_idx,
            )
            eval_groups = await art.gather_trajectory_groups(
                [
                    art.TrajectoryGroup(
                        vla_rollout_wrapped(scenario)
                        for _ in range(1)
                    )
                ],
                pbar_desc=f"eval {i + 1}/{len(eval_scenarios)}",
            )
            for g in eval_groups:
                for t in g.trajectories:
                    t.metadata["eval_scenario_idx"] = i
                    all_eval_trajectories.append(t)

        eval_elapsed = _time.time() - t_eval_start
        logger.info(
            "Evaluation done: %d trajectories in %.1fs",
            len(all_eval_trajectories), eval_elapsed,
        )

        current_step = await attack_model.get_step()
        eval_metrics = compute_metrics_from_trajectories(all_eval_trajectories)
        _print_metrics(eval_metrics, step=current_step, logger_fn=logger.info)

        _eval_suite_tag = "+".join(_eval_suite_names)
        latex_label = f"{_eval_suite_tag} (step {current_step})"
        logger.info("LaTeX table row:\n%s", metrics_to_latex_row(eval_metrics, label=latex_label))

        _eval_report_name = (
            f"eval_step{current_step}_{_eval_suite_tag}"
            f"_n{len(all_eval_trajectories)}.json"
        )
        report = {
            "config": {
                "run_name": _run_name,
                "train_suites": args.task_suite,
                "train_suite_specs": {sn: ids for sn, ids in suite_specs},
                "eval_suites": _eval_suite_str,
                "eval_suite_specs": {sn: ids for sn, ids in eval_suite_specs},
                "eval_episodes_per_task": args.eval_episodes_per_task,
                "checkpoint_step": current_step,
                "vla_config": args.vla_config_name,
                "vla_checkpoint": args.vla_checkpoint,
                "base_model": args.base_model,
                "elapsed_seconds": eval_elapsed,
            },
            "summary": eval_metrics.to_dict(),
            "per_task": {
                k: {
                    "count": s.count,
                    "task_execution_rate": s.task_execution_rate,
                    "attack_success_rate": s.attack_success_rate,
                    "avg_reward": s.avg_reward,
                    "avg_tools_called": s.avg_tools_called,
                    "avg_chars_changed": s.avg_chars_changed,
                    "avg_action_seq_length": s.avg_action_seq_length,
                    "avg_constraint_violations": s.avg_constraint_violations,
                }
                for k, s in eval_metrics.per_task.items()
            },
            "per_episode": [
                {
                    "scenario_idx": t.metadata.get("eval_scenario_idx", i),
                    "task_suite": t.metadata.get("task_suite", ""),
                    "task_id": t.metadata.get("task_id", -1),
                    "episode_idx": t.metadata.get("episode_idx", -1),
                    "reward": t.reward,
                    "baseline_success": t.metrics.get("baseline_success", 0),
                    "attack_success": t.metrics.get("attack_success", 0),
                    "num_tool_calls": t.metrics.get("num_tool_calls", 0),
                }
                for i, t in enumerate(all_eval_trajectories)
            ],
        }

        # Save to the timestamped run directory
        _run_eval_path = os.path.join(_run_dir, _eval_report_name)
        with open(_run_eval_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Eval report saved to %s", _run_eval_path)

        # Also save to the flat eval_reports/ for easy browsing
        _flat_report_dir = os.path.join(_ART_OUTPUT_ROOT, "eval_reports")
        os.makedirs(_flat_report_dir, exist_ok=True)
        _flat_path = os.path.join(_flat_report_dir, _eval_report_name)
        with open(_flat_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Eval report also saved to %s", _flat_path)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a VLA adversarial attack agent via GRPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Attack config ---
    parser.add_argument(
        "--objective", type=str, default=DEFAULT_OBJECTIVE,
        choices=[o.value for o in AttackObjective],
        help="Attack objective for this training run.",
    )
    parser.add_argument(
        "--tool_sets", type=str, default=DEFAULT_TOOL_SETS,
        help="Comma-separated tool families: token,char,prompt,visual. Use several so the agent can deploy multi-tool attacks.",
    )
    parser.add_argument(
        "--max_turns", type=int, default=8,
        help="Max ReAct tool-call rounds per episode. Increase (e.g. 8–12) so the agent can chain multiple tools for stronger attacks. Default: 8.",
    )
    parser.add_argument(
        "--stealth_weight", type=float, default=DEFAULT_STEALTH_WEIGHT,
        help="λ for the stealth penalty in the reward.",
    )
    parser.add_argument(
        "--no_attack_penalty", type=float, default=-1.0,
        help="Fixed reward when the agent does not call any attack tool (default -1.0). Stronger than -0.5 to encourage tool use.",
    )
    parser.add_argument(
        "--short_trajectory_penalty", type=float, default=0.2,
        help="Extra penalty when post-attack trajectory is much shorter than baseline (default 0.2). Set 0 to disable.",
    )
    parser.add_argument(
        "--short_trajectory_ratio", type=float, default=0.5,
        help="Apply short_trajectory_penalty when attack_steps < baseline_steps * this (default 0.5 = 50%% shorter).",
    )
    parser.add_argument(
        "--max_edit_chars", type=int, default=DEFAULT_MAX_EDIT_CHARS,
        help="Hard budget: max Levenshtein character edits (add/remove/change) allowed. Applies to all tool types.",
    )
    parser.add_argument(
        "--replan_steps", type=int, default=DEFAULT_REPLAN_STEPS,
        help=(
            "Actions to execute from each VLA prediction chunk before re-planning. "
            "Higher = fewer VLA model calls (faster). Default 5: matches official "
            "OpenPI LIBERO config. Range 5–50."
        ),
    )

    # --- LIBERO task config ---
    parser.add_argument(
        "--task_suite", type=str, default=DEFAULT_TASK_SUITE,
        help=(
            "LIBERO task suite(s). Comma-separated for multi-suite. "
            "Default: 4 eval suites (40 tasks). "
            "Valid: libero_spatial, libero_object, libero_goal, libero_10, libero_90."
        ),
    )
    parser.add_argument(
        "--task_ids", type=str, default=DEFAULT_TASK_IDS,
        help=(
            "Task indices within the suite.  Supports: 'all', ranges like "
            "'0-89', or comma-separated '0,5,10-19'."
        ),
    )
    parser.add_argument(
        "--episodes_per_task", type=int, default=DEFAULT_EPISODES_PER_TASK,
        help="Number of initial states (episodes) per task.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Max steps per episode. Default 800 for action_inflation (headroom so attack can succeed with more steps).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Environment seed (use 42 to match eval.run_libero_eval baseline).")

    # --- Post-training evaluation config ---
    parser.add_argument(
        "--eval_task_suite", type=str, default=None,
        help=(
            "LIBERO suite(s) for post-training evaluation. Comma-separated. "
            "Defaults to the same suites used for training."
        ),
    )
    parser.add_argument(
        "--eval_task_ids", type=str, default=DEFAULT_EVAL_TASK_IDS,
        help=(
            "Task indices for evaluation (default: 7-9, held-out test split). "
            "Same syntax as --task_ids."
        ),
    )
    parser.add_argument(
        "--eval_episodes_per_task", type=int, default=DEFAULT_EVAL_EPISODES_PER_TASK,
        help="Initial states per test task. More = more robust metrics. Default: 10.",
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip automatic post-training evaluation on the eval suite.",
    )

    # --- VLA model config ---
    parser.add_argument(
        "--vla_config_name", type=str, default="pi05_libero",
        help="OpenPI training config name for Pi0.5.",
    )
    parser.add_argument(
        "--vla_checkpoint", type=str, default=None,
        help="Path to Pi0.5 checkpoint (None = auto-download base model).",
    )

    # --- GPU layout (default: GPU 0,1,2 = VLA rollouts, GPU 3 = agent) --
    parser.add_argument(
        "--vla_gpus", type=str,
        default=os.environ.get("VLA_GPUS", os.environ.get("VLA_GPU", "0,1,2")),
        help=(
            "Comma-separated GPU indices for Pi0.5 VLA model(s) (JAX).  "
            "One model is loaded per GPU; rollouts use a round-robin pool for "
            "parallel multi-GPU inference.  Default: '0,1,2' → 3 models."
        ),
    )
    parser.add_argument(
        "--attack_gpus", type=str, default=_ATTACK_GPUS,
        help=(
            "GPU index (or comma-separated list) for the vLLM attack agent.  "
            "Default: all visible GPUs except --vla_gpus (e.g. '1,2,3' on 4 GPUs).  "
            "vLLM runs with tensor_parallel_size=1, so only the first GPU in this list is used."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.65,
        help=(
            "Fraction of GPU memory vLLM may use on each attack GPU.  "
            "If the model-service subprocess is killed (e.g. signal 6 / OOM), try 0.5 or 0.45; "
            "for single-GPU (VLA and attack on same GPU) use 0.4–0.5."
        ),
    )

    # --- ART / training config ---
    parser.add_argument(
        "--model_name", type=str, default="qwen2.5-3B",
        help="Name of the attack agent model for ART.",
    )
    parser.add_argument(
        "--project_name", type=str, default="vla-attack-agent",
        help="ART project name.",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace base model for the attack agent (default: Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing checkpoints for this model. Default is to train "
            "from scratch (previous checkpoints are cleared). Use --resume only when "
            "you explicitly want to continue a previous run."
        ),
    )
    parser.add_argument(
        "--trajectories_per_group", type=int,
        default=DEFAULT_TRAJECTORIES_PER_GROUP,
        help="Number of rollouts per GRPO group.",
    )
    parser.add_argument(
        "--groups_per_step", type=int, default=DEFAULT_GROUPS_PER_STEP,
        help="Number of groups to gather before each training step.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs over the scenario set.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="Learning rate for GRPO.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=DEFAULT_EVAL_STEPS,
        help="Run evaluation every N training steps.",
    )
    parser.add_argument(
        "--rollout_workers", type=int, default=24,
        help=(
            "Number of concurrent VLA rollout episodes.  Each worker runs "
            "a rollout in its own thread: VLA inference is serialised per GPU "
            "via a lock, while MuJoCo steps run concurrently on CPU.  "
            "With multiple --vla_gpus, workers are assigned round-robin.  "
            "Default 24 = 8 per GPU with 3 VLA GPUs (A100-80GB)."
        ),
    )
    args = parser.parse_args()

    try:
        asyncio.run(train(args))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
