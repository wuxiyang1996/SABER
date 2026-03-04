#!/usr/bin/env python3
"""Baseline LIBERO evaluation for OpenVLA using model_factory.

Evaluates OpenVLA (per-suite finetuned checkpoints) on LIBERO suites without
any attack. Records per-task and per-suite success rates and saves a JSON report.

Key differences from DeepThinkVLA eval:
  - action_horizon=1 (single action per inference, no chunking)
  - replan_steps=1 (replan every step)
  - Per-suite checkpoints: model must be reloaded for each suite

Usage:
    python eval_baseline_openvla.py --gpu 1 --suites libero_spatial,libero_object,libero_goal,libero_10
    python eval_baseline_openvla.py --gpu 1 --task_ids 7-9  # held-out test tasks only
    python eval_baseline_openvla.py --gpu 1 --task_ids 0-9 --suites libero_spatial
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib
import logging
from datetime import datetime

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTHONUTF8", "1")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from PIL import Image

logger = logging.getLogger("eval_baseline_openvla")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_task_ids(s):
    ids = []
    for part in s.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def create_libero_env(task_suite_name, task_id, seed, resolution=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

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


def reset_env(env, initial_states, episode_idx=0):
    env.reset()
    init_state = initial_states[episode_idx % len(initial_states)]
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def preprocess_image_openvla(img, size=224):
    """Match the official OpenVLA RLDS training pipeline preprocessing."""
    import io as _io
    img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img)
    buf = _io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf).convert("RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def run_episode(env, initial_states, vla_model, instruction, episode_idx,
                max_steps, replan_steps):
    from libero_rollouts.pi05_libero_model import build_libero_state

    obs = reset_env(env, initial_states, episode_idx)
    action_buffer = []
    total_steps = 0
    success = False

    vla_model.set_language(instruction)

    for step in range(max_steps):
        if not action_buffer:
            agentview = preprocess_image_openvla(obs["agentview_image"])
            wrist = preprocess_image_openvla(obs["robot0_eye_in_hand_image"])
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


def main():
    parser = argparse.ArgumentParser(
        description="Baseline LIBERO evaluation for OpenVLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--suites", type=str,
                        default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--task_ids", type=str, default="7-9")
    parser.add_argument("--episodes_per_task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan_steps", type=int, default=1,
                        help="OpenVLA has action_horizon=1, so replan every step.")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_eval")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    suite_names = [s.strip() for s in args.suites.split(",")]
    task_ids = parse_task_ids(args.task_ids)

    logger.info("=" * 60)
    logger.info("OpenVLA Baseline Evaluation")
    logger.info("=" * 60)
    logger.info("  Suites:         %s", suite_names)
    logger.info("  Task IDs:       %s", task_ids)
    logger.info("  Episodes/task:  %d", args.episodes_per_task)
    logger.info("  Seed:           %d", args.seed)
    logger.info("  Replan steps:   %d", args.replan_steps)
    logger.info("  GPU:            %s", args.gpu)

    from libero_rollouts.model_factory import load_vla_model, get_checkpoint_for_suite

    results = {}
    all_successes = []
    per_suite = {}
    t0 = time.time()

    for suite_name in suite_names:
        logger.info("\n--- Suite: %s ---", suite_name)
        max_steps = MAX_STEPS.get(suite_name, 300)

        ckpt = get_checkpoint_for_suite("openvla", suite_name)
        logger.info("  Checkpoint: %s", ckpt)

        vla_model = load_vla_model(
            "openvla",
            suite_name=suite_name,
            device="cuda:0",
            replan_steps=args.replan_steps,
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
                    ep, max_steps, args.replan_steps,
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

    report = {
        "model": "openvla",
        "checkpoints": {
            sn: get_checkpoint_for_suite("openvla", sn)
            for sn in suite_names
        },
        "config": {
            "suites": suite_names,
            "task_ids": task_ids,
            "episodes_per_task": args.episodes_per_task,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
            "action_horizon": 1,
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
    report_path = os.path.join(args.output_dir, f"baseline_openvla_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
