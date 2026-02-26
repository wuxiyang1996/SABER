"""Post-training evaluation of the VLA adversarial attack agent.

Evaluates how effectively the trained attack agent degrades the frozen
Pi0.5 VLA policy on LIBERO tasks.  Runs attacked rollouts on a set of
evaluation tasks and reports all paper metrics.

Usage
-----
::

    cd agent_attack_framework

    # Evaluate on libero_10 (standard eval set, default):
    python eval_vla_attack.py \\
        --vla_gpus 0 --attack_gpus 3

    # Evaluate on a specific subset:
    python eval_vla_attack.py \\
        --task_suite libero_spatial --task_ids 0-4 \\
        --episodes_per_task 5 --trajectories_per_group 1

Outputs a detailed JSON report and prints all metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Ensure project root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_LIBERO_DIR = os.path.join(_THIS_DIR, "libero_rollouts")
if _LIBERO_DIR not in sys.path:
    sys.path.insert(0, _LIBERO_DIR)
_TOOLS_DIR = os.path.join(_THIS_DIR, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("eval_vla_attack")

import art
from art.langgraph import wrap_rollout

from agent.vla_rollout import (
    AttackObjective,
    ToolSet,
    VLAAttackScenario,
    build_scenarios,
    clear_baseline_cache,
    parse_task_ids,
    set_vla_models,
    vla_attack_rollout,
)
from rwd_func.metrics import (
    compute_metrics_from_trajectories,
    print_metrics,
    metrics_to_latex_row,
)


_ART_OUTPUT_ROOT = os.path.join(_THIS_DIR, "outputs")


async def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation."""
    import concurrent.futures

    loop = asyncio.get_running_loop()
    pool_size = max(args.rollout_workers * 2, 8)
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=pool_size),
    )

    objective = AttackObjective(args.objective)
    tool_sets = [ToolSet(t.strip()) for t in args.tool_sets.split(",")]
    task_ids = parse_task_ids(args.task_ids, args.task_suite)

    logger.info("=" * 60)
    logger.info("VLA Attack Agent — Post-Training Evaluation")
    logger.info("=" * 60)
    logger.info("  Objective:        %s", objective.value)
    logger.info("  Tool sets:        %s", [ts.value for ts in tool_sets])
    logger.info("  Task suite:       %s", args.task_suite)
    logger.info("  Task IDs:         %s", task_ids)
    logger.info("  Episodes/task:    %d", args.episodes_per_task)
    logger.info("  Trajs/group:      %d", args.trajectories_per_group)

    # --- Load VLA model(s) ---
    import jax
    jax_devices = jax.devices("gpu")

    vla_gpus = [int(g.strip()) for g in args.vla_gpus.split(",")]
    from pi05_libero_model import Pi05LiberoModel

    models_and_devices = []
    for i, vla_gpu in enumerate(vla_gpus):
        jax_idx = i
        if jax_idx >= len(jax_devices):
            raise RuntimeError(
                f"VLA GPU {vla_gpu} mapped to JAX idx {jax_idx} but only "
                f"{len(jax_devices)} GPU(s) visible."
            )
        dev = jax_devices[jax_idx]
        logger.info("Loading Pi0.5 on JAX device %s (GPU %d)...", dev, vla_gpu)
        with jax.default_device(dev):
            model = Pi05LiberoModel(
                train_config_name=args.vla_config_name,
                checkpoint_path=args.vla_checkpoint,
                action_horizon=50,
                replan_steps=5,
            )
        models_and_devices.append((model, dev))
    set_vla_models(models_and_devices)
    logger.info("VLA model pool: %d instance(s)", len(models_and_devices))

    # --- Switch to attack GPUs for vLLM ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.attack_gpus
    logger.info("CUDA_VISIBLE_DEVICES=%s for vLLM", args.attack_gpus)

    # --- ART backend ---
    from art.local.backend import LocalBackend

    attack_model = art.TrainableModel(
        name=args.model_name,
        project=args.project_name,
        base_model=args.base_model,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=4096),
            engine_args=art.dev.EngineArgs(
                gpu_memory_utilization=args.gpu_memory_utilization,
            ),
        ),
    )

    backend = LocalBackend(path=_ART_OUTPUT_ROOT)
    await attack_model.register(backend)
    current_step = await attack_model.get_step()
    logger.info("Loaded attack model checkpoint at step %d", current_step)

    # --- Build eval scenarios ---
    eval_scenarios = build_scenarios(
        objective=objective,
        tool_sets=tool_sets,
        task_suite_name=args.task_suite,
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        stealth_weight=args.stealth_weight,
        max_turns=args.max_turns,
        replan_steps=args.replan_steps,
        seed=args.seed,
    )
    logger.info("Built %d evaluation scenarios.", len(eval_scenarios))

    # --- Run evaluation rollouts ---
    vla_rollout_wrapped = wrap_rollout(
        attack_model,
        lambda scenario: vla_attack_rollout(attack_model, scenario),
    )

    all_trajectories = []
    t0 = time.time()

    for i, scenario in enumerate(eval_scenarios):
        clear_baseline_cache()
        logger.info(
            "Scenario %d/%d: %s task %d, ep %d ...",
            i + 1, len(eval_scenarios),
            scenario.task_suite_name, scenario.task_id,
            scenario.episode_idx,
        )

        groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    vla_rollout_wrapped(scenario)
                    for _ in range(args.trajectories_per_group)
                )
            ],
            pbar_desc=f"eval {i+1}/{len(eval_scenarios)}",
        )

        for g in groups:
            for t in g.trajectories:
                t.metadata["eval_scenario_idx"] = i
                all_trajectories.append(t)

    elapsed = time.time() - t0
    logger.info("Evaluation done: %d trajectories in %.1fs", len(all_trajectories), elapsed)

    # --- Compute and print metrics ---
    metrics = compute_metrics_from_trajectories(all_trajectories)
    text = print_metrics(metrics, step=current_step, logger_fn=logger.info)
    print()
    print(text)

    # LaTeX row for paper table
    print()
    latex_label = f"{args.task_suite} (step {current_step})"
    print("LaTeX table row:")
    print(metrics_to_latex_row(metrics, label=latex_label))

    # --- Save detailed JSON report ---
    report_dir = os.path.join(
        _ART_OUTPUT_ROOT, "eval_reports",
    )
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(
        report_dir,
        f"eval_step{current_step}_{args.task_suite}_n{len(all_trajectories)}.json",
    )

    report = {
        "config": {
            "objective": objective.value,
            "tool_sets": [ts.value for ts in tool_sets],
            "task_suite": args.task_suite,
            "task_ids": task_ids,
            "episodes_per_task": args.episodes_per_task,
            "trajectories_per_group": args.trajectories_per_group,
            "checkpoint_step": current_step,
            "base_model": args.base_model,
            "elapsed_seconds": elapsed,
        },
        "summary": metrics.to_dict(),
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
            for k, s in metrics.per_task.items()
        },
        "per_episode": [
            {
                "scenario_idx": t.metadata.get("eval_scenario_idx", i),
                "task_suite": t.metadata.get("task_suite", ""),
                "task_id": t.metadata.get("task_id", -1),
                "episode_idx": t.metadata.get("episode_idx", -1),
                "objective": t.metadata.get("objective", ""),
                "reward": t.reward,
                "baseline_success": t.metrics.get("baseline_success", 0),
                "attack_success": t.metrics.get("attack_success", 0),
                "baseline_steps": t.metrics.get("baseline_steps", 0),
                "attack_steps": t.metrics.get("attack_steps", 0),
                "num_tool_calls": t.metrics.get("num_tool_calls", 0),
                "tools_used": t.metadata.get("tools_used", ""),
                "instruction_changed": t.metrics.get("instruction_changed", 0),
                "observation_changed": t.metrics.get("observation_changed", 0),
                "original_instruction": t.metadata.get("original_instruction", ""),
                "perturbed_instruction": t.metadata.get("perturbed_instruction", ""),
                "stealth_penalty": t.metrics.get("stealth_penalty", 0.0),
                "stealth_text_edit_ratio": t.metrics.get("stealth_text_edit_ratio", 0.0),
                "stealth_char_edit_dist": t.metrics.get("stealth_char_edit_dist", 0.0),
                "collision_count": t.metrics.get("attack_collision_count", 0),
                "joint_limit_violations": t.metrics.get("attack_joint_limit_violations", 0),
                "excessive_force_count": t.metrics.get("attack_excessive_force_count", 0),
            }
            for i, t in enumerate(all_trajectories)
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Detailed report saved to %s", report_path)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained VLA attack agent on LIBERO tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--objective", type=str, default="task_failure",
                        choices=[o.value for o in AttackObjective])
    parser.add_argument("--tool_sets", type=str, default="token,char,prompt")
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--task_ids", type=str, default="all",
                        help=(
                            "Task indices.  Supports: 'all', ranges like "
                            "'0-9', or comma-separated '0,3,5-7'."
                        ))
    parser.add_argument("--episodes_per_task", type=int, default=3)
    parser.add_argument("--trajectories_per_group", type=int, default=1,
                        help="Rollouts per scenario (1 for deterministic eval).")
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--stealth_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--vla_config_name", type=str, default="pi05_libero")
    parser.add_argument("--vla_checkpoint", type=str, default=None)
    parser.add_argument("--vla_gpus", type=str, default="0,1,2")
    parser.add_argument("--attack_gpus", type=str, default="3")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)

    parser.add_argument("--model_name", type=str, default="qwen2.5-3B")
    parser.add_argument("--project_name", type=str, default="vla-attack-agent")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--rollout_workers", type=int, default=4)

    args = parser.parse_args()

    try:
        asyncio.run(evaluate(args))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted.")
    except Exception:
        logger.exception("Evaluation failed.")
        raise


if __name__ == "__main__":
    main()
