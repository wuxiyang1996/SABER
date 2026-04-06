#!/usr/bin/env python3
"""
Eval all 6 paper VLAs on LIBERO 4 suites in parallel across 4 GPUs.

- Assigns 2-3 models per GPU (configurable via --models_per_gpu); each GPU runs
  its models sequentially so we load and run multiple models on the same GPU.
- For native models: runs eval with parallel env threads over CPU cores.
- For external models: runs their eval script in subprocess (requires --repo_paths).
- Aggregates metrics into a single performance report.

Usage (from agent_attack_framework):
    python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3
    python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --models_per_gpu 3
    python -m eval.run_all_libero_evals_parallel --gpus 0,1,2,3 --models openpi_pi05,openvla,ecot --cpu_workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from eval.model_registry import NATIVE_MODELS, get_model_defaults
from eval.external.configs import get_external_config

# GitHub URLs for auto-clone when --repo_paths is not set (clone to cache dir)
_EXTERNAL_REPO_GITHUB = {
    "openvla": "https://github.com/openvla/openvla",
    "molmoact": "https://github.com/allenai/molmoact",
    "deepthinkvla": "https://github.com/OpenBMB/DeepThinkVLA",
    "ecot": "https://github.com/openvla/openvla",
    "internvla_m1": "https://github.com/InternRobotics/InternVLA-M1",
}


def _repos_dir() -> str:
    """Local repos directory: agent_attack_framework/repos/<model_id>."""
    return os.path.join(_FRAMEWORK_ROOT, "repos")


def _repos_cache_dir() -> str:
    return os.path.join(
        os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache")),
        "repos",
    )


def _resolve_repo_path_from_dirs(model_id: str) -> Optional[str]:
    """
    If a repo exists in repos/<model_id> (or for ecot, repos/ecot or repos/openvla), return it.
    Used so you can clone all repos into agent_attack_framework/repos/ and skip --repo_path.
    """
    repos_root = _repos_dir()
    for candidate in (model_id, "openvla" if model_id == "ecot" else None):
        if candidate is None:
            continue
        path = os.path.join(repos_root, candidate)
        if os.path.isdir(path):
            return path
    return None


def _ensure_repo_available(model_id: str) -> Optional[str]:
    """
    Resolve repo path: first repos/<model_id>, then .cache/repos clone, then clone from GitHub.
    Returns None if clone fails or model has no cloneable repo.
    """
    path = _resolve_repo_path_from_dirs(model_id)
    if path is not None:
        return path
    url = _EXTERNAL_REPO_GITHUB.get(model_id)
    if not url or url.startswith("("):
        return None
    cache_root = _repos_cache_dir()
    repo_dir = os.path.join(cache_root, model_id)
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        return repo_dir
    try:
        os.makedirs(cache_root, exist_ok=True)
        print(f"  Auto-cloning {model_id} from {url} to {repo_dir}...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, repo_dir],
            check=True,
            capture_output=True,
            timeout=300,
            cwd="/",
        )
        return repo_dir
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"  Auto-clone failed for {model_id}: {e}", flush=True)
        return None


ALL_MODELS = list(NATIVE_MODELS) + [
    "openvla", "molmoact", "deepthinkvla", "ecot", "internvla_m1",
]
DEFAULT_SUITES = "spatial,object,goal,long"
DEFAULT_TASK_IDS = "0-9"
DEFAULT_EPISODES_PER_TASK = 5
DEFAULT_SEED = 42


def parse_task_ids(s: str) -> List[int]:
    ids = []
    for part in s.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def assign_models_to_gpus(
    model_ids: List[str],
    gpus: List[int],
    max_models_per_gpu: int,
) -> List[Tuple[int, List[str]]]:
    """
    Assign models to GPUs so each GPU runs 2-3 models (or up to max_models_per_gpu).
    Distributes as evenly as possible: e.g. 10 models on 4 GPUs -> [3,3,2,2].
    Returns list of (gpu_id, [model_id, ...]) for each GPU that has at least one model.
    """
    n = len(model_ids)
    k = len(gpus)
    if n == 0 or k == 0:
        return []
    # Target: high = ceil(n/k), low = floor(n/k); first (n % k) GPUs get high, rest get low
    high = (n + k - 1) // k
    low = n // k
    high = min(high, max_models_per_gpu)
    low = min(low, max_models_per_gpu) if low > 0 else high
    n_high = n - low * k
    if n_high < 0:
        n_high = 0
    assignments: List[List[str]] = []
    idx = 0
    for i in range(k):
        size = high if i < n_high else low
        if size <= 0 or idx >= n:
            break
        assignments.append(model_ids[idx : idx + size])
        idx += size
    # If we capped by max_models_per_gpu, assign remainder round-robin
    while idx < n:
        for i in range(len(assignments)):
            if idx >= n:
                break
            if len(assignments[i]) < max_models_per_gpu:
                assignments[i].append(model_ids[idx])
                idx += 1
    return [(gpus[i], assignments[i]) for i in range(len(assignments)) if assignments[i]]


def run_one_model_eval(
    model_id: str,
    gpu_id: int,
    repo_paths: Dict[str, str],
    output_dir: str,
    suites: str,
    task_ids: str,
    episodes_per_task: Optional[int],
    seed: Optional[int],
    cpu_workers: int,
    replan_steps: Optional[int],
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """
    Run eval for one model on the given GPU. Returns (model_id, gpu_id, result_json_path, stderr).
    If episodes_per_task, seed, or replan_steps is None, uses get_model_defaults(model_id).
    """
    defaults = get_model_defaults(model_id)
    if episodes_per_task is None:
        episodes_per_task = defaults["episodes_per_task"]
    if seed is None:
        seed = defaults["seed"]
    if replan_steps is None:
        replan_steps = defaults["replan_steps"]
    print(f"  Running {model_id} on GPU {gpu_id}...", flush=True)
    env = os.environ.copy()
    # Use same HF cache as eval.download_checkpoints so checkpoints load automatically
    _cache_root = os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache"))
    env.setdefault("HF_HOME", os.path.join(_cache_root, "huggingface"))
    env.setdefault("HF_HUB_CACHE", os.path.join(_cache_root, "huggingface", "hub"))
    env.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache_root, "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"{model_id}_{ts}.json")
    log_path = os.path.join(output_dir, f"{model_id}_{ts}.log")

    if model_id in NATIVE_MODELS:
        cmd = [
            sys.executable, "-m", "eval.run_libero_eval",
            "--model", model_id,
            "--suites", suites,
            "--task_ids", task_ids,
            "--episodes_per_task", str(episodes_per_task),
            "--seed", str(seed),
            "--replan_steps", str(replan_steps),
            "--output_dir", output_dir,
        ]
        if cpu_workers > 1:
            cmd += ["--parallel_envs", str(cpu_workers)]
        proc = subprocess.run(
            cmd,
            cwd=_FRAMEWORK_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 2,
        )
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            f.write(proc.stderr)
        if proc.returncode != 0:
            return (model_id, gpu_id, None, proc.stderr)
        # Find the newest result json (run_libero_eval writes with its own timestamp)
        candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith(model_id) and f.endswith(".json") and "external" not in f
        ]
        if not candidates:
            return (model_id, gpu_id, None, "No result JSON found")
        best = max(candidates, key=lambda p: os.path.getmtime(p))
        return (model_id, gpu_id, best, None)
    else:
        # External model: use provided repo_path or try to auto-download (clone) the repo
        repo_path = repo_paths.get(model_id, "")
        if not repo_path:
            repo_path = _ensure_repo_available(model_id)
        if not repo_path:
            return (model_id, gpu_id, None, f"Skipped: no repo (use --repo_paths {model_id}=/path to run)")
        cmd = [
            sys.executable, "-m", "eval.external.run",
            "--model", model_id,
            "--episodes_per_task", str(episodes_per_task),
            "--seed", str(seed),
            "--output_dir", output_dir,
        ]
        if repo_path:
            cmd += ["--repo_path", repo_path]
        proc = subprocess.run(
            cmd,
            cwd=_FRAMEWORK_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 3,
        )
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            f.write(proc.stderr)
        if proc.returncode != 0:
            return (model_id, gpu_id, None, proc.stderr)
        candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith(model_id) and "external" in f and f.endswith(".json")
        ]
        if candidates:
            best = max(candidates, key=lambda p: os.path.getmtime(p))
            return (model_id, gpu_id, best, None)
        return (model_id, gpu_id, None, "No result JSON found (check eval/results/<model>_*.log for subprocess output)")


def run_gpu_batch(
    gpu_id: int,
    model_ids: List[str],
    repo_paths: Dict[str, str],
    output_dir: str,
    suites: str,
    task_ids: str,
    episodes_per_task: Optional[int],
    seed: Optional[int],
    cpu_workers: int,
    replan_steps: Optional[int],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Run all assigned models on one GPU sequentially. Returns {model_id: (result_path, error)}.
    """
    results = {}
    for model_id in model_ids:
        _, _, path, err = run_one_model_eval(
            model_id, gpu_id, repo_paths, output_dir,
            suites, task_ids, episodes_per_task, seed,
            cpu_workers, replan_steps,
        )
        results[model_id] = (path, err)
    return results


def load_result_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _per_task_from_data(data: Dict) -> Dict[str, Dict[str, float]]:
    """Build per_task[suite][task_id] = success_rate from result JSON (summary.per_task or per_suite)."""
    per_task: Dict[str, Dict[str, float]] = {}
    summary = data.get("summary") or {}
    if "per_task" in summary and isinstance(summary["per_task"], dict):
        for suite, tasks in summary["per_task"].items():
            if not isinstance(tasks, dict):
                continue
            per_task[suite] = {
                tid: (v.get("success_rate") if isinstance(v, dict) else None)
                for tid, v in tasks.items()
            }
        return per_task
    # Fallback: compute from per_suite[suite][task_id] = list of episode results
    per_suite_raw = data.get("per_suite") or {}
    for suite, tasks in per_suite_raw.items():
        if not isinstance(tasks, dict):
            continue
        per_task[suite] = {}
        for task_id_str, ep_list in tasks.items():
            if not isinstance(ep_list, list):
                continue
            n = len(ep_list)
            s = sum(1 for r in ep_list if r.get("success"))
            per_task[suite][task_id_str] = s / n if n else 0.0
    return per_task


def aggregate_metrics(result_paths: Dict[str, Optional[str]]) -> Dict:
    """Build performance table from result JSONs; includes per_suite and per_task breakdown."""
    rows = []
    for model_id, path in result_paths.items():
        if not path or not os.path.isfile(path):
            rows.append({
                "model": model_id,
                "overall_success_rate": None,
                "per_suite": {},
                "per_task": {},
                "error": "no result file",
            })
            continue
        data = load_result_json(path)
        if not data:
            rows.append({
                "model": model_id,
                "overall_success_rate": None,
                "per_suite": {},
                "per_task": {},
                "error": "failed to load JSON",
            })
            continue
        summary = data.get("summary") or {}
        overall = summary.get("overall") or {}
        per_suite = {
            k: (v.get("success_rate") if isinstance(v, dict) else None)
            for k, v in summary.items()
            if k not in ("overall", "per_task") and isinstance(v, dict)
        }
        per_task = _per_task_from_data(data)
        rows.append({
            "model": model_id,
            "overall_success_rate": overall.get("success_rate"),
            "overall_success": overall.get("success"),
            "overall_episodes": overall.get("episodes"),
            "per_suite": per_suite,
            "per_task": per_task,
            "error": None,
        })
    return {"models": rows, "result_paths": {k: v for k, v in result_paths.items() if v}}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eval all 6 paper VLAs on LIBERO in parallel across 4 GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model ids or 'all'",
    )
    parser.add_argument("--suites", type=str, default=DEFAULT_SUITES)
    parser.add_argument("--task_ids", type=str, default=DEFAULT_TASK_IDS)
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=None,
        help="Episodes per task (default: per-model from MODEL_DEFAULT_HYPERPARAMS, usually 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: per-model from MODEL_DEFAULT_HYPERPARAMS, usually 42)",
    )
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=4,
        help="Parallel LIBERO/MuJoCo env threads per native model",
    )
    parser.add_argument(
        "--repo_paths",
        type=str,
        default="",
        help="Comma-separated key=value e.g. openvla=/path. If omitted, OpenVLA/MolmoAct/etc. are auto-cloned to .cache/repos/<model> (requires git). X-VLA needs no repo (uses lerobot-eval from HF).",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=None,
        help="Replan steps for native models (default: per-model from MODEL_DEFAULT_HYPERPARAMS, usually 5)",
    )
    parser.add_argument(
        "--models_per_gpu",
        type=int,
        default=3,
        help="Max models to run sequentially per GPU (2-3 typical)",
    )
    args = parser.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",")]
    if not gpus:
        print("At least one GPU required (--gpus 0,1,2,3)", file=sys.stderr)
        return 1

    if args.models.strip().lower() == "all":
        model_ids = list(ALL_MODELS)
    else:
        model_ids = [x.strip().lower().replace("-", "_") for x in args.models.split(",")]
        for m in model_ids:
            if m not in ALL_MODELS:
                print(f"Unknown model: {m}", file=sys.stderr)
                return 1

    repo_paths = {}
    for part in args.repo_paths.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            repo_paths[k.strip().lower().replace("-", "_")] = v.strip()

    output_dir = args.output_dir or os.path.join(_FRAMEWORK_ROOT, "eval", "results")
    os.makedirs(output_dir, exist_ok=True)

    gpu_assignments = assign_models_to_gpus(model_ids, gpus, args.models_per_gpu)
    print(f"GPUs: {gpus}")
    print(f"Models: {model_ids}")
    print(f"Models per GPU (max): {args.models_per_gpu}")
    for gid, mids in gpu_assignments:
        print(f"  GPU {gid}: {mids}")
    print(f"CPU workers per native model: {args.cpu_workers}")
    print(f"Output: {output_dir}")

    result_paths: Dict[str, Optional[str]] = {m: None for m in model_ids}
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(gpu_assignments)) as ex:
        futures = {
            ex.submit(
                run_gpu_batch,
                gpu_id,
                mids,
                repo_paths,
                output_dir,
                args.suites,
                args.task_ids,
                args.episodes_per_task,
                args.seed,
                args.cpu_workers,
                args.replan_steps,
            ): (gpu_id, mids)
            for gpu_id, mids in gpu_assignments
        }
        for fut in as_completed(futures):
            gpu_id, mids = futures[fut]
            try:
                batch_results = fut.result()
                for mid, (path, err) in batch_results.items():
                    result_paths[mid] = path
                    if err:
                        errors[mid] = err
                        if err.startswith("Skipped:"):
                            print(f"  {mid} (GPU {gpu_id}): {err[:70]}")
                        else:
                            print(f"  {mid} (GPU {gpu_id}): error - {err[:80]}")
                    else:
                        if path:
                            print(f"  {mid} (GPU {gpu_id}): done -> {path}")
                        else:
                            print(f"  {mid} (GPU {gpu_id}): done but no result JSON (check eval/results/{mid}_*.log)")
                            errors[mid] = "No result JSON found"
            except Exception as e:
                for mid in mids:
                    errors[mid] = str(e)
                    print(f"  {mid}: exception - {e}")

    # Aggregate metrics
    report = aggregate_metrics(result_paths)
    report["errors"] = errors
    report["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report_path = os.path.join(output_dir, f"all_models_metrics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nMetrics written to {report_path}")

    # Print overall + per-suite table
    print("\n--- Performance ---")
    print(f"{'Model':<20} {'Overall':<10} {'Spatial':<10} {'Object':<10} {'Goal':<10} {'Long':<10}")
    print("-" * 70)
    for row in report["models"]:
        m = row["model"]
        o = row.get("overall_success_rate")
        ps = row.get("per_suite") or {}
        def pct(v):
            return f"{v*100:.1f}%" if v is not None else "N/A"
        o_str = pct(o)
        s1 = pct(ps.get("libero_spatial"))
        s2 = pct(ps.get("libero_object"))
        s3 = pct(ps.get("libero_goal"))
        s4 = pct(ps.get("libero_10"))
        print(f"{m:<20} {o_str:<10} {s1:<10} {s2:<10} {s3:<10} {s4:<10}")

    # Per-task breakdown by category (suite)
    suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    suite_short = {"libero_spatial": "spatial", "libero_object": "object", "libero_goal": "goal", "libero_10": "long"}
    for suite in suite_order:
        task_ids_sorted = []
        for row in report["models"]:
            pt = row.get("per_task") or {}
            tasks = pt.get(suite) or {}
            for tid in tasks:
                if tid not in task_ids_sorted:
                    task_ids_sorted.append(tid)
        try:
            task_ids_sorted.sort(key=int)
        except (ValueError, TypeError):
            pass
        if not task_ids_sorted:
            continue
        short = suite_short.get(suite, suite)
        print(f"\n--- Per-task: {short} ({suite}) ---")
        col_w = 5
        header = f"{'Model':<20}" + "".join(f"{str(tid):>{col_w}}" for tid in task_ids_sorted)
        print(header)
        print("-" * (20 + col_w * len(task_ids_sorted)))
        for row in report["models"]:
            pt = row.get("per_task") or {}
            tasks = pt.get(suite) or {}
            line = f"{row['model']:<20}"
            for tid in task_ids_sorted:
                v = tasks.get(str(tid), tasks.get(tid))
                line += f"{v*100:.0f}%".rjust(col_w) if v is not None else "N/A".rjust(col_w)
            print(line)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
