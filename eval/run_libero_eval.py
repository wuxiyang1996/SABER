#!/usr/bin/env python3
"""
Unified LIBERO evaluation on the 4 evaluation suites (spatial, object, goal, long).

Runs a single VLA model on all tasks in the specified suites with the same
episode counts and seed for comparable results. Outputs JSON to eval/results/.

Usage (from agent_attack_framework):
    python -m eval.run_libero_eval --model openpi_pi05 --suites spatial,object,goal,long
    python -m eval.run_libero_eval --model openpi_pi0 --episodes_per_task 5 --seed 42

For external models (OpenVLA, X-VLA, MolmoAct, etc.) use --model <id> and
optionally --repo_path; they are run via eval.external.run. Native models:
openpi_pi0, openpi_pi05.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Ensure framework root is on path when run as python -m eval.run_libero_eval
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

# Use same HF cache as eval.download_checkpoints so checkpoints load from cache automatically
_EVAL_CACHE_ROOT = os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_EVAL_CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_EVAL_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_EVAL_CACHE_ROOT, "huggingface"))

from eval.model_registry import NATIVE_MODELS, get_model_defaults


def _create_libero_env(task_suite_name: str, task_id: int, seed: int, resolution: int = 256):
    # Uses libero + rwd_func (no model)
    """Create LIBERO environment for a task (same logic as agent.vla_rollout)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, initial_states, task.language


def _reset_env(env, initial_states, episode_idx: int = 0) -> dict:
    """Reset env to initial state and stabilise."""
    env.reset()
    init_state = initial_states[episode_idx % len(initial_states)]
    obs = env.set_init_state(init_state)
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)
    return obs


def run_episode(env, initial_states, policy_fn, instruction: str, episode_idx: int, max_steps: int):
    """Run one episode and return (success, num_steps)."""
    from rwd_func.rwd import collect_libero_rollout_info
    obs = _reset_env(env, initial_states, episode_idx)
    base_env = env.env if hasattr(env, "env") else env
    info = collect_libero_rollout_info(
        env=base_env,
        policy_fn=policy_fn,
        instruction=instruction,
        observation=obs,
        max_steps=max_steps,
    )
    return info.task_success, info.num_steps


def main():
    parser = argparse.ArgumentParser(
        description="Run LIBERO evaluation for a VLA model on 4 suites (spatial/object/goal/long).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_models = list(NATIVE_MODELS) + [
        "openvla", "xvla", "molmoact", "deepthinkvla", "ecot", "internvla_m1", "starvla", "lightvla",
    ]
    parser.add_argument(
        "--model",
        type=str,
        default="openpi_pi05",
        help="Model: openpi_pi0, openpi_pi05, or external (openvla, xvla, molmoact, deepthinkvla, ecot, internvla_m1, starvla, lightvla).",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="spatial,object,goal,long",
        help="Comma-separated suite names: spatial, object, goal, long (or libero_*)",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default="0-9",
        help="Task indices per suite (e.g. 0-9 for all 10, or 7-9 for test only).",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=5,
        help="Number of initial states (episodes) per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for env and initial state selection.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path for OpenPI models (optional).",
    )
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=5,
        help="Actions executed per inference chunk (replan interval).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for result JSON (default: eval/results under framework root).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="LIBERO camera resolution.",
    )
    parser.add_argument(
        "--parallel_envs",
        type=int,
        default=1,
        help="Number of parallel LIBERO env threads (CPU). >1 uses parallel_episode_runner.",
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default=None,
        help="For external models: path to model repo (e.g. OpenVLA clone).",
    )
    parser.add_argument(
        "--print_cmd",
        action="store_true",
        help="For external models: only print eval commands, do not run.",
    )
    args = parser.parse_args()

    model_id = args.model.lower().replace("-", "_")
    defaults = get_model_defaults(model_id)
    if args.episodes_per_task is None:
        args.episodes_per_task = defaults["episodes_per_task"]
    if args.seed is None:
        args.seed = defaults["seed"]
    if args.replan_steps is None:
        args.replan_steps = defaults["replan_steps"]

    from eval.external.configs import get_external_config
    if get_external_config(model_id) is not None:
        # Dispatch to external runner; use repos/<model_id> if present and --repo_path not set
        from eval.external.run import get_commands, run_commands, resolve_repo_path_from_dirs
        args.repo_path = args.repo_path or resolve_repo_path_from_dirs(model_id)
        if not args.print_cmd and model_id not in ("xvla", "starvla") and not args.repo_path:
            print("For external model '%s' provide --repo_path or put the repo in agent_attack_framework/repos/%s." % (model_id, model_id), file=sys.stderr)
            return 1
        commands = get_commands(
            model_id,
            repo_path=args.repo_path,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
        )
        if not commands:
            print("No commands for this external model. See eval/external/configs.py.", file=sys.stderr)
            return 1
        if args.print_cmd:
            for c in commands:
                print(c)
            return 0
        return run_commands(
            commands,
            print_only=False,
            output_dir=args.output_dir or os.path.join(_FRAMEWORK_ROOT, "eval", "results"),
            model_id=model_id,
        )

    if model_id not in NATIVE_MODELS:
        print(f"Unknown model: {args.model}. Native: {list(NATIVE_MODELS)}. External: openvla, xvla, molmoact, deepthinkvla, ecot, internvla_m1, starvla, lightvla.", file=sys.stderr)
        return 1

    # Native models: --print_cmd just confirms the model and suite layout (no model load)
    if args.print_cmd:
        print(f"Native model: {model_id}")
        print(f"Suites: spatial,object,goal,long (libero_spatial, libero_object, libero_goal, libero_10)")
        print(f"Run without --print_cmd to execute eval (requires openpi + LIBERO).")
        return 0

    # Resolve suite names and task ids (needed for both paths)
    from eval.model_registry import SUITE_ALIASES
    suite_names = []
    for s in args.suites.split(","):
        s = s.strip().lower()
        suite_names.append(SUITE_ALIASES.get(s, s))
    if not suite_names:
        suite_names = list(SUITE_ALIASES.values())[:4]
    task_ids = []
    for part in args.task_ids.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            task_ids.extend(range(int(lo), int(hi) + 1))
        else:
            task_ids.append(int(part))
    task_ids = sorted(set(task_ids))

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(_FRAMEWORK_ROOT, "eval", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Parallel env path (multiple CPU workers)
    if getattr(args, "parallel_envs", 1) > 1:
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip())
        from eval.parallel_episode_runner import run_native_parallel
        run_native_parallel(
            model_id=model_id,
            suite_names=suite_names,
            task_ids=task_ids,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
            replan_steps=args.replan_steps,
            resolution=getattr(args, "resolution", 256),
            n_parallel_envs=args.parallel_envs,
            gpu_id=gpu_id,
            output_path=os.path.join(output_dir, f"{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"),
        )
        return 0

    # EGL/headless before any MuJoCo/OpenGL
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    from eval.model_registry import MAX_STEPS, load_model, make_policy_fn
    from rwd_func.rwd import collect_libero_rollout_info

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(
        args.model,
        checkpoint_path=args.checkpoint,
        replan_steps=args.replan_steps,
    )

    results = {
        "model": args.model,
        "suites": suite_names,
        "task_ids": task_ids,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_suite": {},
        "summary": {},
    }

    total_episodes = 0
    total_success = 0

    for suite_name in suite_names:
        max_steps = MAX_STEPS.get(suite_name, 300)
        results["per_suite"][suite_name] = {}
        results["summary"][suite_name] = {"success": 0, "episodes": 0}

        for task_id in task_ids:
            try:
                env, initial_states, instruction = _create_libero_env(
                    suite_name, task_id, args.seed, args.resolution
                )
            except Exception as e:
                print(f"[WARN] {suite_name} task {task_id}: env creation failed: {e}")
                results["per_suite"][suite_name][str(task_id)] = []
                continue

            policy_fn = make_policy_fn(model, instruction, args.replan_steps)
            task_results = []

            for ep in range(args.episodes_per_task):
                success, num_steps = run_episode(
                    env, initial_states, policy_fn, instruction, ep, max_steps
                )
                task_results.append({"episode_idx": ep, "success": success, "num_steps": num_steps})
                results["summary"][suite_name]["episodes"] += 1
                total_episodes += 1
                if success:
                    results["summary"][suite_name]["success"] += 1
                    total_success += 1

            env.close()
            results["per_suite"][suite_name][str(task_id)] = task_results
            n_ok = sum(1 for r in task_results if r["success"])
            print(f"  {suite_name} task {task_id}: {n_ok}/{len(task_results)} success")

        n = results["summary"][suite_name]["episodes"]
        s = results["summary"][suite_name]["success"]
        results["summary"][suite_name]["success_rate"] = s / n if n else 0.0
        print(f"  {suite_name} total: {s}/{n} = {100 * results['summary'][suite_name]['success_rate']:.1f}%")

    # Per-task breakdown for reference (success/episodes/success_rate per (suite, task_id))
    results["summary"]["per_task"] = {}
    for suite_name in suite_names:
        results["summary"]["per_task"][suite_name] = {}
        for task_id_str, task_results in results["per_suite"].get(suite_name, {}).items():
            if not isinstance(task_results, list):
                continue
            n_ep = len(task_results)
            n_ok = sum(1 for r in task_results if r.get("success"))
            results["summary"]["per_task"][suite_name][task_id_str] = {
                "success": n_ok,
                "episodes": n_ep,
                "success_rate": n_ok / n_ep if n_ep else 0.0,
            }

    results["summary"]["overall"] = {
        "success": total_success,
        "episodes": total_episodes,
        "success_rate": total_success / total_episodes if total_episodes else 0.0,
    }

    out_name = f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
