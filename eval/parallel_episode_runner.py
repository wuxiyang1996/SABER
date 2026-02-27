"""
Run LIBERO episodes in parallel over CPU cores: N env threads, 1 model (serialized).

Used by run_libero_eval when --parallel_envs N > 1. Each thread runs one env;
model inference is serialized with a lock so we get CPU parallelism for
MuJoCo stepping while only one inference at a time.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# Framework root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def run_native_parallel(
    model_id: str,
    suite_names: List[str],
    task_ids: List[int],
    episodes_per_task: int,
    seed: int,
    replan_steps: int,
    resolution: int,
    n_parallel_envs: int,
    gpu_id: int,
    output_path: str,
) -> Dict[str, Any]:
    """
    Run native model eval with n_parallel_envs env threads.
    Model runs on gpu_id; inference is serialized; env stepping runs in parallel on CPU.
    """
    import json
    from eval.model_registry import MAX_STEPS, load_model, make_policy_fn
    from eval.run_libero_eval import _create_libero_env, _reset_env
    from rwd_func.rwd import collect_libero_rollout_info

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = load_model(model_id, replan_steps=replan_steps)
    model_lock = threading.Lock()

    def run_one_episode(suite_name: str, task_id: int, ep_idx: int) -> Tuple[str, int, int, bool, int]:
        max_steps = MAX_STEPS.get(suite_name, 300)
        try:
            env, initial_states, instruction = _create_libero_env(suite_name, task_id, seed, resolution)
        except Exception:
            return (suite_name, task_id, ep_idx, False, -1)
        policy_fn = make_policy_fn(model, instruction, replan_steps)
        obs = _reset_env(env, initial_states, ep_idx)
        base_env = env.env if hasattr(env, "env") else env
        with model_lock:
            info = collect_libero_rollout_info(
                env=base_env, policy_fn=policy_fn, instruction=instruction,
                observation=obs, max_steps=max_steps,
            )
        env.close()
        return (suite_name, task_id, ep_idx, info.task_success, info.num_steps)

    work_items = [
        (suite, tid, ep)
        for suite in suite_names
        for tid in task_ids
        for ep in range(episodes_per_task)
    ]
    n_workers = min(n_parallel_envs, len(work_items), 8)
    results_list: List[Tuple[str, int, int, bool, int]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one_episode, s, t, e): (s, t, e) for (s, t, e) in work_items}
        for fut in as_completed(futs):
            try:
                results_list.append(fut.result())
            except Exception:
                s, t, e = futs[fut]
                results_list.append((s, t, e, False, -1))

    results_by_suite: Dict[str, Dict[str, List[Dict]]] = {s: {} for s in suite_names}
    summary: Dict[str, Dict] = {s: {"success": 0, "episodes": 0} for s in suite_names}
    for (suite, task_id, ep_idx, success, num_steps) in results_list:
        if suite not in results_by_suite:
            continue
        key = str(task_id)
        if key not in results_by_suite[suite]:
            results_by_suite[suite][key] = []
        results_by_suite[suite][key].append({"episode_idx": ep_idx, "success": success, "num_steps": num_steps})
        summary[suite]["episodes"] += 1
        if success:
            summary[suite]["success"] += 1
    for suite in suite_names:
        n = summary[suite]["episodes"]
        s = summary[suite]["success"]
        summary[suite]["success_rate"] = s / n if n else 0.0
    total_success = sum(summary[s]["success"] for s in suite_names)
    total_episodes = sum(summary[s]["episodes"] for s in suite_names)
    # Per-task breakdown for reference
    per_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for suite in suite_names:
        per_task[suite] = {}
        for task_id_str, ep_list in results_by_suite.get(suite, {}).items():
            if not isinstance(ep_list, list):
                continue
            n_ep = len(ep_list)
            n_ok = sum(1 for r in ep_list if r.get("success"))
            per_task[suite][task_id_str] = {
                "success": n_ok,
                "episodes": n_ep,
                "success_rate": n_ok / n_ep if n_ep else 0.0,
            }
    report = {
        "model": model_id,
        "suites": suite_names,
        "task_ids": task_ids,
        "episodes_per_task": episodes_per_task,
        "seed": seed,
        "n_parallel_envs": n_parallel_envs,
        "gpu_id": gpu_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_suite": results_by_suite,
        "summary": {
            **summary,
            "per_task": per_task,
            "overall": {
                "success": total_success,
                "episodes": total_episodes,
                "success_rate": total_success / total_episodes if total_episodes else 0,
            },
        },
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
