#!/usr/bin/env python3
"""Unified baseline LIBERO evaluation for VLA models.

Evaluates a VLA model on LIBERO suites without any attack.  Records per-task
and per-suite success rates and saves a JSON report.

Model-specific behaviour (replan_steps default, image preprocessing) is
handled via MODEL_DEFAULTS so adding a new model only requires one dict entry.

Usage:
    python eval_baseline_vla.py --model openvla   --gpu 1
    python eval_baseline_vla.py --model deepthinkvla --gpu 0
    python eval_baseline_vla.py --model openvla --task_ids 7-9 --suites libero_spatial
    python eval_baseline_vla.py --model ecot --gpu 2 --suites libero_spatial,libero_object
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONUTF8", "1")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from libero_utils import MAX_STEPS, parse_task_ids, create_libero_env, reset_env

logger = logging.getLogger("eval_baseline_vla")


# ---------------------------------------------------------------------------
# Model-specific defaults
# ---------------------------------------------------------------------------

MODEL_DEFAULTS = {
    "openvla": {
        "replan_steps": 1,
        "preprocess": "openvla",
        "action_horizon": 1,
    },
    "deepthinkvla": {
        "replan_steps": 5,
        "preprocess": "standard",
    },
    "ecot": {
        "replan_steps": 1,
        "preprocess": "standard",
        "action_horizon": 1,
    },
    "lightvla": {
        "replan_steps": 1,
        "preprocess": "standard",
        "action_horizon": 1,
    },
}


def _get_preprocess_fn(model_id: str):
    """Return the image preprocessing function for the given model."""
    preprocess_type = MODEL_DEFAULTS.get(model_id, {}).get("preprocess", "standard")
    if preprocess_type == "openvla":
        from libero_rollouts.openvla_wrapper import preprocess_image_openvla
        return preprocess_image_openvla
    from libero_rollouts.pi05_libero_model import preprocess_image
    return preprocess_image


def _get_default_replan(model_id: str) -> int:
    return MODEL_DEFAULTS.get(model_id, {}).get("replan_steps", 5)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, initial_states, vla_model, instruction, episode_idx,
                max_steps, replan_steps, preprocess_fn):
    from libero_rollouts.pi05_libero_model import build_libero_state

    obs = reset_env(env, initial_states, episode_idx)
    action_buffer = []
    total_steps = 0
    success = False

    vla_model.set_language(instruction)

    for step in range(max_steps):
        if not action_buffer:
            agentview = preprocess_fn(obs["agentview_image"])
            wrist = preprocess_fn(obs["robot0_eye_in_hand_image"])
            state = build_libero_state(obs)
            actions = vla_model.predict(agentview, wrist, state)
            action_buffer.extend(actions[:replan_steps].tolist())

        action = np.array(action_buffer.pop(0), dtype=np.float64)
        obs, reward, done, info = env.step(action.tolist())
        total_steps += 1

        if done or info.get("success", False) or reward > 0:
            success = True
            break

    return success, total_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline LIBERO evaluation for VLA models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="openvla",
                        help="VLA model id (openvla, deepthinkvla, ecot, lightvla, …).")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--suites", type=str,
                        default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--task_ids", type=str, default="7-9")
    parser.add_argument("--episodes_per_task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan_steps", type=int, default=None,
                        help="Override replan_steps (auto-detected per model if omitted).")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_eval")
    args = parser.parse_args()

    model_id = args.model.lower().replace("-", "_")
    replan_steps = args.replan_steps if args.replan_steps is not None else _get_default_replan(model_id)
    preprocess_fn = _get_preprocess_fn(model_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    suite_names = [s.strip() for s in args.suites.split(",")]
    task_ids = parse_task_ids(args.task_ids)

    logger.info("=" * 60)
    logger.info("%s Baseline Evaluation", model_id)
    logger.info("=" * 60)
    logger.info("  Model:          %s", model_id)
    logger.info("  Suites:         %s", suite_names)
    logger.info("  Task IDs:       %s", task_ids)
    logger.info("  Episodes/task:  %d", args.episodes_per_task)
    logger.info("  Seed:           %d", args.seed)
    logger.info("  Replan steps:   %d", replan_steps)
    logger.info("  GPU:            %s", args.gpu)

    from libero_rollouts.model_factory import load_vla_model, get_checkpoint_for_suite

    all_successes = []
    per_suite = {}
    t0 = time.time()

    for suite_name in suite_names:
        logger.info("\n--- Suite: %s ---", suite_name)
        max_steps = MAX_STEPS.get(suite_name, 300)

        try:
            ckpt = get_checkpoint_for_suite(model_id, suite_name)
            logger.info("  Checkpoint: %s", ckpt)
        except ValueError:
            ckpt = None

        vla_model = load_vla_model(
            model_id,
            suite_name=suite_name,
            device="cuda:0",
            replan_steps=replan_steps,
        )

        suite_successes = []
        task_results = {}

        for task_id in task_ids:
            env, initial_states, instruction = create_libero_env(
                suite_name, task_id, args.seed,
            )
            logger.info("  Task %d: %s", task_id, instruction)

            ep_results = []
            for ep in range(args.episodes_per_task):
                success, steps = run_episode(
                    env, initial_states, vla_model, instruction,
                    ep, max_steps, replan_steps, preprocess_fn,
                )
                ep_results.append({"success": success, "steps": steps})
                logger.info("    Episode %d: success=%s, steps=%d",
                            ep, success, steps)

            sr = sum(1 for r in ep_results if r["success"]) / len(ep_results)
            task_results[task_id] = {
                "instruction": instruction,
                "episodes": ep_results,
                "success_rate": sr,
            }
            suite_successes.extend([r["success"] for r in ep_results])
            logger.info("    Task %d SR: %.1f%%", task_id, sr * 100)

            try:
                env.close()
            except Exception:
                pass

        suite_sr = sum(suite_successes) / len(suite_successes) if suite_successes else 0
        per_suite[suite_name] = {
            "success_rate": suite_sr,
            "num_episodes": len(suite_successes),
            "num_successes": sum(suite_successes),
            "tasks": task_results,
        }
        all_successes.extend(suite_successes)
        logger.info("  Suite %s SR: %.1f%% (%d/%d)",
                     suite_name, suite_sr * 100,
                     sum(suite_successes), len(suite_successes))

        if hasattr(vla_model, "shutdown"):
            vla_model.shutdown()
        del vla_model
        import torch
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    overall_sr = sum(all_successes) / len(all_successes) if all_successes else 0

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for sn, data in per_suite.items():
        logger.info("  %s: %.1f%% (%d/%d)",
                     sn, data["success_rate"] * 100,
                     data["num_successes"], data["num_episodes"])
    logger.info("  OVERALL: %.1f%% (%d/%d)",
                 overall_sr * 100, sum(all_successes), len(all_successes))
    logger.info("  Time: %.1fs", elapsed)

    # Build checkpoint info for the report
    checkpoints = {}
    for sn in suite_names:
        try:
            checkpoints[sn] = get_checkpoint_for_suite(model_id, sn)
        except ValueError:
            pass

    report = {
        "model": model_id,
        "checkpoints": checkpoints,
        "config": {
            "suites": suite_names,
            "task_ids": task_ids,
            "episodes_per_task": args.episodes_per_task,
            "seed": args.seed,
            "replan_steps": replan_steps,
            "gpu": args.gpu,
        },
        "overall_success_rate": overall_sr,
        "num_episodes": len(all_successes),
        "num_successes": sum(all_successes),
        "per_suite": per_suite,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output_dir, f"baseline_{model_id}_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
