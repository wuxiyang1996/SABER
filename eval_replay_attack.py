#!/usr/bin/env python3
"""Replay pre-recorded attack prompts on any victim VLA model.

Loads perturbed instructions from a previously-generated attack record
(JSON from eval_attack_vla.py / run_record_agent_outputs_task_failure.sh)
and evaluates each VLA on both the original and perturbed instruction,
WITHOUT running the attack agent.  This is much faster (no LLM inference)
and allows measuring transferability of attacks across models.

Inputs
------
A source JSON file (``--attack_record``) produced by eval_attack_vla.py.
Each entry in ``per_episode`` contains:
    original_instruction, perturbed_instruction,
    task_suite, task_id, episode_idx, tools_used, ...

The script runs the victim VLA twice per episode:
    1. Baseline rollout with ``original_instruction``
    2. Attack rollout with ``perturbed_instruction``

Outputs
-------
A JSON report in ``--output_dir`` with the same structure as
eval_attack_vla.py output, plus a ``source_record`` field.

Usage
-----
    # Evaluate pre-recorded openpi_pi05 attacks on OpenVLA
    python eval_replay_attack.py \\
        --victim openvla \\
        --attack_record outputs/agent_output_records_task_failure_2/task_failure_openpi_pi05.json \\
        --vla_gpu 0 \\
        --output_dir outputs/replay_eval_task_failure

    # Use the vla_models conda env for HF-based VLAs
    VLA_PYTHON=/workspace/miniforge3/envs/vla_models/bin/python \\
    python eval_replay_attack.py --victim openvla ...
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--victim", type=str, default="openvla")
_pre_parser.add_argument("--vla_gpu", type=str, default="0")
_pre_args, _ = _pre_parser.parse_known_args()

_IS_JAX_VICTIM = _pre_args.victim.lower().replace("-", "_") in ("openpi_pi0", "openpi_pi05")
_VLA_GPU = str(_pre_args.vla_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = _VLA_GPU

if _IS_JAX_VICTIM:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONUTF8", "1")

_CACHE_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".cache"))
os.environ.setdefault("OPENPI_DATA_HOME", _CACHE_ROOT)
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE_ROOT, "huggingface"))

import json
import logging
import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import numpy as np

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

logger = logging.getLogger("eval_replay_attack")

from libero_utils import (
    MAX_STEPS as _MAX_STEPS,
    create_libero_env as _create_libero_env,
    run_vla_episode,
)


def load_attack_record(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_replay(args):
    from rwd_func.rwd import edit_distance
    from libero_rollouts.model_factory import load_vla_model, is_jax_model, is_per_suite_model

    victim_id = args.victim.lower().replace("-", "_")
    vla_is_jax = is_jax_model(victim_id)
    vla_per_suite = is_per_suite_model(victim_id)
    vla_lock = threading.Lock()

    record = load_attack_record(args.attack_record)
    episodes = record["per_episode"]
    source_victim = record["config"]["victim_model"]
    source_objective = record["config"]["objective"]

    logger.info("=" * 60)
    logger.info("Replay Attack Evaluation")
    logger.info("=" * 60)
    logger.info("  Victim VLA:       %s", victim_id)
    logger.info("  Source record:    %s", args.attack_record)
    logger.info("  Source victim:    %s", source_victim)
    logger.info("  Source objective: %s", source_objective)
    episodes_per_task = getattr(args, "episodes_per_task", None)
    logger.info("  Record entries:  %d", len(episodes))
    if episodes_per_task:
        logger.info("  Episodes/task:   %d (override)", episodes_per_task)
    logger.info("  Seed:            %d", args.seed)

    num_workers = getattr(args, "num_workers", 1) or 1
    worker_id = getattr(args, "worker_id", 0) or 0
    fast_mode = getattr(args, "fast", False)
    if num_workers > 1:
        logger.info("  Worker:          %d / %d", worker_id, num_workers)
    if fast_mode:
        logger.info("  Fast mode:       ON (skip predicates/snapshots/obs storage)")

    # Filter episodes for this worker
    if num_workers > 1:
        episodes = [ep for i, ep in enumerate(episodes) if i % num_workers == worker_id]
        logger.info("  Episodes (this worker): %d", len(episodes))

    # Load JAX model once (not per-suite)
    vla_model = None
    if vla_is_jax:
        import jax
        jax_device = jax.devices("gpu")[0]
        logger.info("Loading JAX VLA model on %s ...", jax_device)
        with jax.default_device(jax_device):
            vla_model = load_vla_model(
                victim_id,
                checkpoint_path=args.vla_checkpoint,
                replan_steps=args.replan_steps,
            )
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_state = np.zeros(8, dtype=np.float32)
        vla_model.set_language("warmup")
        with jax.default_device(jax_device):
            _ = vla_model.predict(dummy_img, dummy_img, dummy_state)
        logger.info("JAX VLA warmup done.")

    # Group episodes by suite for efficient model loading
    suite_episodes = defaultdict(list)
    for ep in episodes:
        suite_episodes[ep["task_suite"]].append(ep)

    per_episode_results = []
    t0 = time.time()
    ep_multiplier = episodes_per_task if episodes_per_task else 1
    total = len(episodes) * ep_multiplier
    done = 0
    current_suite = None

    for suite_name in sorted(suite_episodes.keys()):
        suite_eps = suite_episodes[suite_name]
        max_steps = args.max_steps if args.max_steps is not None else _MAX_STEPS.get(suite_name, 300)

        # Load/reload model for this suite
        if not vla_is_jax:
            if vla_per_suite or vla_model is None or current_suite != suite_name:
                logger.info("Loading %s VLA for suite %s ...", victim_id, suite_name)
                try:
                    if vla_model is not None:
                        if hasattr(vla_model, "shutdown"):
                            vla_model.shutdown()
                        del vla_model
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                vla_model = load_vla_model(
                    victim_id,
                    suite_name=suite_name,
                    checkpoint_path=args.vla_checkpoint,
                    device="cuda:0",
                    replan_steps=args.replan_steps,
                )
                current_suite = suite_name

        for ep in suite_eps:
            task_id = ep["task_id"]
            original_instruction = ep["original_instruction"]
            perturbed_instruction = ep["perturbed_instruction"]
            source_tools = ep.get("tools_used", [])

            ep_indices = list(range(episodes_per_task)) if episodes_per_task else [ep["episode_idx"]]

            for episode_idx in ep_indices:
                logger.info(
                    "  [%d/%d] %s task %d ep %d ...",
                    done + 1, total, suite_name, task_id, episode_idx,
                )

                env, initial_states, libero_instruction = _create_libero_env(
                    suite_name, task_id, args.seed,
                )

                if libero_instruction != original_instruction:
                    logger.warning(
                        "    Instruction mismatch for %s task %d: "
                        "record='%s' vs libero='%s'",
                        suite_name, task_id, original_instruction, libero_instruction,
                    )

                baseline_info = run_vla_episode(
                    env, initial_states, vla_model,
                    original_instruction, episode_idx, max_steps,
                    args.replan_steps, is_jax=vla_is_jax, vla_lock=vla_lock,
                    fast=fast_mode,
                )

                attack_info = run_vla_episode(
                    env, initial_states, vla_model,
                    perturbed_instruction, episode_idx, max_steps,
                    args.replan_steps, is_jax=vla_is_jax, vla_lock=vla_lock,
                    fast=fast_mode,
                )

                chars_changed = int(edit_distance(original_instruction, perturbed_instruction))
                flipped = baseline_info.task_success and not attack_info.task_success

                bl_col = float(baseline_info.collision_count)
                bl_jl = float(baseline_info.joint_limit_violations)
                bl_ef = float(baseline_info.excessive_force_count)
                atk_col = float(attack_info.collision_count)
                atk_jl = float(attack_info.joint_limit_violations)
                atk_ef = float(attack_info.excessive_force_count)

                result = {
                    "task_suite": suite_name,
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "original_instruction": original_instruction,
                    "perturbed_instruction": perturbed_instruction,
                    "tools_used": source_tools,
                    "num_tool_calls": len(source_tools),
                    "chars_changed": chars_changed,
                    "flipped": flipped,
                    "baseline": {
                        "success": bool(baseline_info.task_success),
                        "steps": int(baseline_info.num_steps),
                        "constraint_violations": bl_col + bl_jl + bl_ef,
                        "collisions": bl_col,
                        "joint_limit_violations": bl_jl,
                        "excessive_force": bl_ef,
                    },
                    "attack": {
                        "success": bool(attack_info.task_success),
                        "steps": int(attack_info.num_steps),
                        "constraint_violations": atk_col + atk_jl + atk_ef,
                        "collisions": atk_col,
                        "joint_limit_violations": atk_jl,
                        "excessive_force": atk_ef,
                    },
                }
                per_episode_results.append(result)

                logger.info(
                    "    baseline=%s(%d) attack=%s(%d) flipped=%s",
                    "PASS" if baseline_info.task_success else "FAIL",
                    baseline_info.num_steps,
                    "PASS" if attack_info.task_success else "FAIL",
                    attack_info.num_steps,
                    "YES" if flipped else "no",
                )

                try:
                    env.close()
                except Exception:
                    pass

                done += 1

    elapsed = time.time() - t0
    logger.info("Replay evaluation done: %d episodes in %.1fs", total, elapsed)

    # --- Compute aggregate metrics ---
    n = len(per_episode_results)
    bl_successes = [r["baseline"]["success"] for r in per_episode_results]
    atk_successes = [r["attack"]["success"] for r in per_episode_results]
    flips = [r["flipped"] for r in per_episode_results]
    bl_steps = [r["baseline"]["steps"] for r in per_episode_results]
    atk_steps = [r["attack"]["steps"] for r in per_episode_results]
    chars = [r["chars_changed"] for r in per_episode_results]
    n_tools = [r["num_tool_calls"] for r in per_episode_results]

    bl_cv = [r["baseline"]["constraint_violations"] for r in per_episode_results]
    atk_cv = [r["attack"]["constraint_violations"] for r in per_episode_results]

    n_baseline_success = sum(bl_successes)
    n_flipped = sum(flips)
    asr = n_flipped / n_baseline_success if n_baseline_success > 0 else 0.0

    step_ratios = []
    for r in per_episode_results:
        if r["baseline"]["steps"] > 0:
            step_ratios.append(r["attack"]["steps"] / r["baseline"]["steps"])

    baseline_summary = {
        "num_episodes": n,
        "task_execution_rate": sum(bl_successes) / n if n else 0.0,
        "avg_action_seq_length": float(np.mean(bl_steps)) if bl_steps else 0.0,
        "std_action_seq_length": float(np.std(bl_steps)) if bl_steps else 0.0,
        "avg_constraint_violations": float(np.mean(bl_cv)) if bl_cv else 0.0,
    }

    attack_summary = {
        "num_episodes": n,
        "task_execution_rate": sum(atk_successes) / n if n else 0.0,
        "avg_action_seq_length": float(np.mean(atk_steps)) if atk_steps else 0.0,
        "std_action_seq_length": float(np.std(atk_steps)) if atk_steps else 0.0,
        "avg_constraint_violations": float(np.mean(atk_cv)) if atk_cv else 0.0,
        "attack_success_rate": asr,
        "num_baseline_success": n_baseline_success,
        "num_flipped": n_flipped,
        "avg_chars_changed": float(np.mean(chars)) if chars else 0.0,
        "avg_tools_called": float(np.mean(n_tools)) if n_tools else 0.0,
        "avg_step_ratio": float(np.mean(step_ratios)) if step_ratios else 0.0,
        "baseline_success_rate": n_baseline_success / n if n else 0.0,
    }

    comparison = {
        "action_seq_length_delta": attack_summary["avg_action_seq_length"] - baseline_summary["avg_action_seq_length"],
        "avg_step_ratio": attack_summary["avg_step_ratio"],
        "task_execution_rate_delta": attack_summary["task_execution_rate"] - baseline_summary["task_execution_rate"],
        "constraint_violations_delta": attack_summary["avg_constraint_violations"] - baseline_summary["avg_constraint_violations"],
        "attack_success_rate": asr,
        "num_flipped": n_flipped,
        "num_baseline_success": n_baseline_success,
    }

    # Per-task breakdown
    per_task = defaultdict(list)
    for r in per_episode_results:
        key = f"{r['task_suite']}/task_{r['task_id']}"
        per_task[key].append(r)

    per_task_summary = {}
    for key, entries in sorted(per_task.items()):
        tc = len(entries)
        bs = sum(1 for e in entries if e["baseline"]["success"])
        fl = sum(1 for e in entries if e["flipped"])
        per_task_summary[key] = {
            "count": tc,
            "task_execution_rate": sum(1 for e in entries if e["attack"]["success"]) / tc,
            "attack_success_rate": fl / bs if bs else 0.0,
            "baseline_success_rate": bs / tc,
            "avg_chars_changed": float(np.mean([e["chars_changed"] for e in entries])),
            "avg_baseline_steps": float(np.mean([e["baseline"]["steps"] for e in entries])),
            "avg_attack_steps": float(np.mean([e["attack"]["steps"] for e in entries])),
        }

    # --- Save report ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = os.path.splitext(os.path.basename(args.attack_record))[0]
    if num_workers > 1:
        report_name = f"replay_{source_objective}_{victim_id}_from_{source_victim}_w{worker_id}.json"
    else:
        report_name = f"replay_{source_objective}_{victim_id}_from_{source_victim}.json"

    report = {
        "config": {
            "victim_model": victim_id,
            "source_record": args.attack_record,
            "source_victim": source_victim,
            "source_objective": source_objective,
            "source_attack_model": record["config"].get("attack_model", ""),
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "episodes_per_task": episodes_per_task,
            "elapsed_seconds": elapsed,
            "timestamp": timestamp,
        },
        "baseline_summary": baseline_summary,
        "attack_summary": attack_summary,
        "comparison": comparison,
        "per_task": per_task_summary,
        "per_episode": per_episode_results,
    }

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, report_name)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    # --- Print summary table ---
    _print_replay_summary(report)

    return report


def _print_replay_summary(report: Dict[str, Any]):
    victim = report["config"]["victim_model"]
    source = report["config"]["source_victim"]
    bl = report["baseline_summary"]
    atk = report["attack_summary"]
    comp = report["comparison"]

    lines = [
        "=" * 60,
        f"  Replay Attack Results: {victim} (attacks from {source})",
        "─" * 60,
        f"  Episodes:              {bl['num_episodes']}",
        f"  Baseline TER:          {bl['task_execution_rate']:.1%}",
        f"  Attack TER:            {atk['task_execution_rate']:.1%}",
        f"  TER delta:             {comp['task_execution_rate_delta']:+.1%}",
        f"  Attack Success Rate:   {atk['attack_success_rate']:.1%} ({atk['num_flipped']}/{atk['num_baseline_success']} flipped)",
        "─" * 60,
        f"  Avg baseline steps:    {bl['avg_action_seq_length']:.1f}",
        f"  Avg attack steps:      {atk['avg_action_seq_length']:.1f}",
        f"  Step ratio delta:      {comp['action_seq_length_delta']:+.1f}",
        f"  Avg chars changed:     {atk['avg_chars_changed']:.1f}",
        "─" * 60,
        "  Per-task breakdown:",
        f"    {'Task':<28} {'N':>3} {'BL-TER':>7} {'ATK-TER':>8} {'ASR':>6}",
    ]

    for key in sorted(report["per_task"].keys()):
        s = report["per_task"][key]
        lines.append(
            f"    {key:<28} {s['count']:>3} "
            f"{s['baseline_success_rate']:>6.0%} "
            f"{s['task_execution_rate']:>7.0%} "
            f"{s['attack_success_rate']:>5.0%}"
        )

    lines.append("=" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Replay pre-recorded attack prompts on a victim VLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--victim", type=str, required=True,
        help="Victim VLA model: openpi_pi0, openpi_pi05, openvla, ecot, "
             "lightvla, deepthinkvla, molmoact, internvla_m1",
    )
    parser.add_argument("--vla_gpu", type=str, default="0")
    parser.add_argument("--vla_checkpoint", type=str, default=None)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument(
        "--attack_record", type=str, required=True,
        help="Path to JSON file with pre-recorded attack episodes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override max episode steps for all suites (official GR00T eval uses 720).",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/replay_eval_task_failure")
    parser.add_argument(
        "--episodes_per_task", type=int, default=None,
        help="Run each perturbed instruction across N initial states (episode_idx 0..N-1). "
             "If not set, uses the episode_idx from the attack record as-is.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Total number of parallel workers. Each processes a 1/N slice of episodes.",
    )
    parser.add_argument(
        "--worker_id", type=int, default=0,
        help="This worker's index (0-based). Processes episodes where index %% num_workers == worker_id.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip predicate collection, scene snapshots, and observation storage for speed.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    evaluate_replay(args)


if __name__ == "__main__":
    main()
