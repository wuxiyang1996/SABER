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

# ---- Model / data cache directory --------------------------------------
# Redirect all model downloads (openpi checkpoints, HuggingFace, etc.)
# to the project directory instead of ~/.cache (which may have limited
# quota on shared clusters).
_PROJECT_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".cache"
)
os.environ.setdefault("OPENPI_DATA_HOME", os.path.realpath(_PROJECT_CACHE))
os.environ.setdefault("HF_HOME", os.path.join(os.path.realpath(_PROJECT_CACHE), "huggingface"))

# ---- Headless rendering (must be set before MuJoCo / PyOpenGL import) --
# On cluster nodes without a display server, MuJoCo must use EGL
# (GPU-accelerated offscreen) or osmesa (CPU software).  EGL is
# strongly preferred when GPUs are available.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# ---- Multi-GPU memory budget ------------------------------------------
# With 4× L40S (48 GB each) we split models across GPUs:
#   GPU 0  →  Pi0.5 VLA model  (JAX, ~6–10 GiB)
#   GPU 1,2,3 → Qwen attack agent (vLLM / ART, 3-way tensor parallel)
#
# IMPORTANT: Do NOT set CUDA_VISIBLE_DEVICES here.
# `import art` triggers `import torch` which initialises the CUDA
# driver.  If only 1 GPU is visible at that point, PyTorch locks in
# num_gpus=1 and later vLLM module imports crash when they try to
# probe device ≥ 1.
#
# Instead we leave all GPUs visible during import time.  JAX is
# pinned to the VLA GPU via jax.device_put (already done in
# vla_rollout.py).  vLLM gets its GPU restriction via
# CUDA_VISIBLE_DEVICES set just before ART spawns its subprocess.

# Default GPU layout (overridable via CLI --vla_gpu / --attack_gpus or env vars).
def _early_resolve_vla_gpu() -> int:
    for i, tok in enumerate(sys.argv):
        if tok == "--vla_gpu" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--vla_gpu="):
            return int(tok.split("=", 1)[1])
    return int(os.environ.get("VLA_GPU", "0"))

_VLA_GPU = _early_resolve_vla_gpu()
_ATTACK_GPUS = os.environ.get("ATTACK_GPUS", "1")

# JAX memory settings — don't pre-allocate; allocate on demand up to 90%.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
from art.utils import iterate_dataset

from agent.vla_rollout import (
    ToolSet,
    VLAAttackScenario,
    set_vla_model,
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
                )
            )
    return scenarios


# ============================================================================
# Main training loop
# ============================================================================

async def train(args: argparse.Namespace) -> None:
    """Run GRPO training for the VLA attack agent."""

    # --- Parse objective and tool sets --------------------------------
    objective = AttackObjective(args.objective)
    tool_sets = [ToolSet(t.strip()) for t in args.tool_sets.split(",")]
    task_ids = [int(t.strip()) for t in args.task_ids.split(",")]

    logger.info("=" * 60)
    logger.info("VLA Attack Agent — GRPO Training")
    logger.info("=" * 60)
    logger.info("  Objective:        %s", objective.value)
    logger.info("  Tool sets:        %s", [ts.value for ts in tool_sets])
    logger.info("  Task suite:       %s", args.task_suite)
    logger.info("  Task IDs:         %s", task_ids)
    logger.info("  Episodes/task:    %d", args.episodes_per_task)
    logger.info("  Trajs/group:      %d", args.trajectories_per_group)
    logger.info("  Groups/step:      %d", args.groups_per_step)
    logger.info("  Epochs:           %d", args.num_epochs)
    logger.info("  LR:               %s", args.learning_rate)
    logger.info("  Stealth weight:   %s", args.stealth_weight)
    logger.info("=" * 60)

    # --- Resolve GPU layout from CLI -----------------------------------
    vla_gpu = args.vla_gpu
    attack_gpus = args.attack_gpus

    logger.info("GPU layout: VLA → GPU %d  |  Attack agent → GPU(s) %s",
                vla_gpu, attack_gpus)

    # --- Load Pi0.5 VLA model (once) on the VLA GPU -------------------
    # All GPUs are visible (we did NOT restrict CUDA_VISIBLE_DEVICES).
    # JAX sees them all; we select the VLA GPU by index.
    import jax
    jax_devices = jax.devices("gpu")
    logger.info(
        "JAX sees %d GPU(s): %s  (VLA will use gpu:%d)",
        len(jax_devices), jax_devices, vla_gpu,
    )
    if vla_gpu >= len(jax_devices):
        raise RuntimeError(
            f"--vla_gpu={vla_gpu} but JAX only sees {len(jax_devices)} GPU(s)"
        )
    vla_jax_device = jax_devices[vla_gpu]

    logger.info("Loading Pi0.5 VLA model on GPU %d ...", vla_gpu)
    # Lazy import to avoid loading heavy deps when just parsing args
    _libero_rollouts_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "libero_rollouts"),
    )
    if _libero_rollouts_dir not in sys.path:
        sys.path.insert(0, _libero_rollouts_dir)
    from pi05_libero_model import Pi05LiberoModel

    # Load VLA model with JAX default device pinned to the VLA GPU
    with jax.default_device(vla_jax_device):
        vla_model = Pi05LiberoModel(
            train_config_name=args.vla_config_name,
            checkpoint_path=args.vla_checkpoint,
            action_horizon=50,
            replan_steps=5,
        )
    set_vla_model(vla_model, jax_device=vla_jax_device)
    logger.info("Pi0.5 VLA model loaded on GPU %d.", vla_gpu)

    # --- Restrict CUDA_VISIBLE_DEVICES for the ART subprocess ---------
    # The parent process's CUDA driver is already initialised with all
    # GPUs visible — changing the env var won't affect it.  But ART's
    # LocalBackend spawns the UnslothService in a *child* process
    # (via mp_actors) which gets a fresh CUDA init and will only see
    # the GPUs listed here.
    os.environ["CUDA_VISIBLE_DEVICES"] = attack_gpus
    logger.info(
        "Set CUDA_VISIBLE_DEVICES=%s for ART subprocess (vLLM).", attack_gpus,
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
        seed=args.seed,
    )
    logger.info("Built %d training scenarios.", len(train_scenarios))

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
    backend = LocalBackend()
    await attack_model.register(backend)

    # --- Training loop ------------------------------------------------
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=args.groups_per_step,
        num_epochs=args.num_epochs,
        initial_step=await attack_model.get_step(),
    )

    for batch in train_iterator:
        logger.info("Step %d — gathering %d trajectory groups ...", batch.step, len(batch.items))

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    vla_attack_rollout(attack_model, scenario)
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
                "Step %d — mean reward: %.3f, min: %.3f, max: %.3f",
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
        help="Comma-separated list of tool families: token,char,prompt,visual.",
    )
    parser.add_argument(
        "--stealth_weight", type=float, default=DEFAULT_STEALTH_WEIGHT,
        help="λ for the stealth penalty in the reward.",
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

    # --- GPU layout ---
    parser.add_argument(
        "--vla_gpu", type=int, default=int(os.environ.get("VLA_GPU", "0")),
        help="GPU index for the Pi0.5 VLA model (JAX).  Default: 0.",
    )
    parser.add_argument(
        "--attack_gpus", type=str, default=os.environ.get("ATTACK_GPUS", "1"),
        help=(
            "Comma-separated GPU indices for the vLLM attack agent.  "
            "Qwen2.5-3B fits on a single GPU; use '1' (default).  "
            "Multi-GPU TP is not supported due to pydantic pickling bug."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.70,
        help=(
            "Fraction of GPU memory vLLM may use on each attack GPU.  "
            "PyTorch/CUDA context can consume ~10 GiB overhead, so 0.70 "
            "is a safe default for 48 GiB L40S GPUs.  Default: 0.70."
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
        help="HuggingFace base model to fine-tune via GRPO.",
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
