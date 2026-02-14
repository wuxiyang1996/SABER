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

# ---- GPU memory budget ------------------------------------------------
# Must be set *before* JAX / PyTorch are imported.
# Pi0.5 (JAX) needs ~6.2 GiB; limit JAX pre-allocation to 25% (~8 GiB)
# so the remaining ~24 GiB is available for vLLM (Qwen attack agent).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

import argparse
import asyncio
import json
import logging

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

    # --- Load Pi0.5 VLA model (once) ----------------------------------
    logger.info("Loading Pi0.5 VLA model ...")
    # Lazy import to avoid loading heavy deps when just parsing args
    _adv_vla_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "adv_agent_vla", "libero_rollouts"),
    )
    if _adv_vla_dir not in sys.path:
        sys.path.insert(0, _adv_vla_dir)
    from pi05_libero_model import Pi05LiberoModel

    vla_model = Pi05LiberoModel(
        train_config_name=args.vla_config_name,
        checkpoint_path=args.vla_checkpoint,
        action_horizon=50,
        replan_steps=5,
    )
    set_vla_model(vla_model)
    logger.info("Pi0.5 VLA model loaded.")

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

    # --- ART model setup ----------------------------------------------
    attack_model = art.TrainableModel(
        name=args.model_name,
        project=args.project_name,
        base_model=args.base_model,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                gpu_memory_utilization=0.6,
                max_seq_length=4096,
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
