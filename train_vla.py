"""GRPO training for the VLA adversarial attack agent.

Trains an LLM attack agent to perturb instructions/observations fed to
π0.5 in LIBERO, optimising for a single declared attack objective via
Group Relative Policy Optimisation (GRPO).

Usage
-----
    python train_vla.py \\
        --objective task_failure \\
        --tool_sets token,char,prompt \\
        --task_suite libero_spatial \\
        --task_ids 0,1,2 \\
        --model_name <your-model> \\
        --stealth_weight 0.3

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

Reference:
  - ART GRPO: https://art.openpipe.ai
  - MCP-RL example: ART/examples/mcp-rl/
"""

from __future__ import annotations

import os
import sys
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
#   GPU 0 (default, or --vla_gpus) →  Pi0.5 VLA model(s) — rollouts (JAX)
#   Remaining GPU(s) (--attack_gpus) →  Attack agent training (vLLM / ART)
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
        raw = os.environ.get("VLA_GPUS", os.environ.get("VLA_GPU", "0"))
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

# JAX memory settings — don't pre-allocate; allocate on demand up to 90%.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
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
    clear_baseline_cache,
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

DEFAULT_OBJECTIVE = "task_failure"
DEFAULT_TOOL_SETS = "token,char,prompt"
DEFAULT_TASK_SUITE = "libero_spatial"
DEFAULT_TASK_IDS = "0"
DEFAULT_EPISODES_PER_TASK = 3        # initial states per task
DEFAULT_TRAJECTORIES_PER_GROUP = 4   # GRPO group size
DEFAULT_GROUPS_PER_STEP = 4          # groups gathered before one train step
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EVAL_STEPS = 5
DEFAULT_STEALTH_WEIGHT = 0.3
DEFAULT_MAX_STEPS = None  # use suite default


# ============================================================================
# Scenario generation
# ============================================================================

def build_scenarios(
    objective: AttackObjective,
    tool_sets: list[ToolSet],
    task_suite_name: str,
    task_ids: list[int],
    episodes_per_task: int,
    stealth_weight: float,
    max_steps: int | None = None,
    max_turns: int = 5,
    replan_steps: int = 10,
    seed: int = 7,
) -> list[VLAAttackScenario]:
    """Build a list of VLAAttackScenario for training.

    Each (task_id, episode_idx) pair produces one scenario.
    """
    scenarios = []
    for tid in task_ids:
        for ep in range(episodes_per_task):
            scenarios.append(
                VLAAttackScenario(
                    task_suite_name=task_suite_name,
                    task_id=tid,
                    episode_idx=ep,
                    seed=seed,
                    objective=objective,
                    tool_sets=list(tool_sets),
                    stealth_weight=stealth_weight,
                    max_steps=max_steps,
                    max_turns=max_turns,
                    replan_steps=replan_steps,
                )
            )
    return scenarios


# ============================================================================
# Main training loop
# ============================================================================

async def train(args: argparse.Namespace) -> None:
    """Run GRPO training for the VLA attack agent."""

    # --- Configure thread pool for parallel VLA rollouts ---------------
    import concurrent.futures
    loop = asyncio.get_running_loop()
    n_workers = getattr(args, "rollout_workers", 4)
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=n_workers),
    )
    logger.info("Thread pool: %d workers for parallel VLA rollouts.", n_workers)

    # --- Parse objective and tool sets --------------------------------
    objective = AttackObjective(args.objective)
    tool_sets = [ToolSet(t.strip()) for t in args.tool_sets.split(",")]
    task_ids = [int(t.strip()) for t in args.task_ids.split(",")]

    logger.info("=" * 60)
    logger.info("VLA Attack Agent — GRPO Training")
    logger.info("=" * 60)
    logger.info("  Objective:        %s", objective.value)
    logger.info("  Tool sets:        %s", [ts.value for ts in tool_sets])
    logger.info("  Max turns:        %d (ReAct tool-call rounds per episode)", args.max_turns)
    logger.info("  Task suite:       %s", args.task_suite)
    logger.info("  Task IDs:         %s", task_ids)
    logger.info("  Episodes/task:    %d", args.episodes_per_task)
    logger.info("  Trajs/group:      %d", args.trajectories_per_group)
    logger.info("  Groups/step:      %d", args.groups_per_step)
    logger.info("  Epochs:           %d", args.num_epochs)
    logger.info("  LR:               %s", args.learning_rate)
    logger.info("  Replan steps:     %d (VLA inference every N env steps)", args.replan_steps)
    logger.info("  Rollout workers:  %d threads", args.rollout_workers)
    logger.info("  Stealth weight:   %s", args.stealth_weight)
    logger.info("=" * 60)

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
                action_horizon=50,
                replan_steps=5,
            )
        models_and_devices.append((vla_model, vla_jax_device))

    set_vla_models(models_and_devices)
    logger.info(
        "Pi0.5 VLA model pool ready: %d instance(s) on GPU(s) %s.",
        len(vla_gpus), vla_gpus,
    )

    # --- Switch CUDA_VISIBLE_DEVICES to attack GPUs for vLLM ----------
    # JAX init only saw VLA GPUs; now expose only the attack GPU(s) so
    # vLLM gets a clean GPU with no JAX memory footprint.
    os.environ["CUDA_VISIBLE_DEVICES"] = _ATTACK_GPUS_PHYSICAL
    logger.info(
        "Set CUDA_VISIBLE_DEVICES=%s for ART subprocess (vLLM).",
        _ATTACK_GPUS_PHYSICAL,
    )

    # --- Build scenarios ----------------------------------------------
    train_scenarios = build_scenarios(
        objective=objective,
        tool_sets=tool_sets,
        task_suite_name=args.task_suite,
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        stealth_weight=args.stealth_weight,
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
    # tensor_parallel_size is left at 1 (default).  Qwen2.5-3B (~6 GiB)
    # fits comfortably on a single L40S.  TP > 1 triggers vLLM
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

    from art.local.backend import LocalBackend
    backend = LocalBackend(path=_ART_OUTPUT_ROOT)
    logger.info("ART output (checkpoints, logs): %s", _ART_OUTPUT_ROOT)
    await attack_model.register(backend)

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

        # Log summary statistics
        rewards = [
            t.reward
            for g in groups
            for t in g.trajectories
        ]
        if rewards:
            logger.info(
                "Step %d — mean reward (attacked rollouts): %.3f, min: %.3f, max: %.3f",
                batch.step,
                sum(rewards) / len(rewards),
                min(rewards),
                max(rewards),
            )

    logger.info("Training complete.")


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
        "--replan_steps", type=int, default=10,
        help=(
            "Actions to execute from each VLA prediction chunk before re-planning. "
            "Higher = fewer VLA model calls per episode (faster) but less reactive. "
            "With action_horizon=50, replan_steps=10 means 1 VLA inference per 10 "
            "env steps (vs. 5 by default). Range 5–20."
        ),
    )

    # --- LIBERO task config ---
    parser.add_argument(
        "--task_suite", type=str, default=DEFAULT_TASK_SUITE,
        choices=["libero_spatial", "libero_object", "libero_goal",
                 "libero_10", "libero_90"],
        help="LIBERO task suite.",
    )
    parser.add_argument(
        "--task_ids", type=str, default=DEFAULT_TASK_IDS,
        help="Comma-separated task indices within the suite.",
    )
    parser.add_argument(
        "--episodes_per_task", type=int, default=DEFAULT_EPISODES_PER_TASK,
        help="Number of initial states (episodes) per task.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Override max steps per episode (None = suite default).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Environment seed.")

    # --- VLA model config ---
    parser.add_argument(
        "--vla_config_name", type=str, default="pi05_libero",
        help="OpenPI training config name for Pi0.5.",
    )
    parser.add_argument(
        "--vla_checkpoint", type=str, default=None,
        help="Path to Pi0.5 checkpoint (None = auto-download base model).",
    )

    # --- GPU layout (default: GPU 0 = VLA rollouts, remaining = agent) --
    parser.add_argument(
        "--vla_gpus", type=str,
        default=os.environ.get("VLA_GPUS", os.environ.get("VLA_GPU", "0")),
        help=(
            "Comma-separated GPU indices for Pi0.5 VLA model(s) (JAX).  "
            "One model copy is loaded per GPU for truly parallel rollouts.  "
            "Default: '0'.  Example: '0,1,2' loads 3 models on GPUs 0-2."
        ),
    )
    parser.add_argument(
        "--attack_gpus", type=str, default=_ATTACK_GPUS,
        help=(
            "GPU index (or comma-separated list) for the vLLM attack agent training.  "
            "Default: all visible GPUs except --vla_gpu (e.g. '1,2,3' on a 4-GPU node).  "
            "GPU 0 is reserved for VLA rollouts."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.65,
        help=(
            "Fraction of GPU memory vLLM may use on each attack GPU.  "
            "Conservative default 0.55 reduces OOM risk; raise to 0.65–0.70 on large GPUs if needed."
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
        "--rollout_workers", type=int, default=4,
        help=(
            "Thread pool size for parallel VLA rollouts.  Each worker runs "
            "a rollout in its own thread (MuJoCo CPU work overlaps, VLA GPU "
            "inference serialises via lock).  4-8 is a good range."
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
