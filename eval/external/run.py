"""
Run LIBERO eval for an external model by executing its repo's eval script(s).

Usage (from agent_attack_framework):
    python -m eval.external.run --model openvla --repo_path /path/to/openvla
    python -m eval.external.run --model xvla --print_cmd   # only print commands
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

# Use same HF cache as eval.download_checkpoints so checkpoints load automatically
_run_cache_root = os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_run_cache_root, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_run_cache_root, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_run_cache_root, "huggingface"))


def _repos_dir() -> str:
    """Local repos directory: agent_attack_framework/repos/<model_id>."""
    return os.path.join(_FRAMEWORK_ROOT, "repos")


def resolve_repo_path_from_dirs(model_id: str) -> Optional[str]:
    """
    If a repo exists in repos/<model_id> (or for ecot, repos/openvla), return it.
    So you can clone all repos into agent_attack_framework/repos/ and skip --repo_path.
    """
    if model_id in ("xvla", "starvla"):
        return None
    repos_root = _repos_dir()
    for candidate in (model_id, "openvla" if model_id == "ecot" else None):
        if candidate is None:
            continue
        path = os.path.join(repos_root, candidate)
        if os.path.isdir(path):
            return path
    return None


from eval.external.configs import (
    DEFAULT_EPISODES_PER_TASK,
    DEFAULT_SEED,
    LIBERO_4_SUITES,
    get_external_config,
)


def _openvla_commands(repo_path: str, episodes: int, seed: int, checkpoint: Optional[str]) -> List[str]:
    """One command per suite for OpenVLA."""
    base = "openvla/openvla-7b-finetuned-{suite}"
    suite_map = {
        "libero_spatial": "libero_spatial",
        "libero_object": "libero_object",
        "libero_goal": "libero_goal",
        "libero_10": "libero_10",
    }
    cmds = []
    for suite in LIBERO_4_SUITES:
        ckpt = checkpoint or base.format(suite=suite)
        cmd = (
            f"cd {repo_path} && python experiments/robot/libero/run_libero_eval.py "
            f"--model_family openvla --pretrained_checkpoint {ckpt} "
            f"--task_suite_name {suite} --center_crop True "
            f"--num_episodes {episodes} --seed {seed}"
        )
        cmds.append(cmd)
    return cmds


def _xvla_commands(repo_path: str, episodes: int, seed: int) -> List[str]:
    """One command per suite for LeRobot X-VLA (lerobot-eval)."""
    cmds = []
    for suite in LIBERO_4_SUITES:
        cmd = (
            f"lerobot-eval --policy.path=lerobot/xvla-libero --env.type=libero --env.task={suite} "
            f"--eval.n_episodes={episodes} --eval.seed={seed}"
        )
        cmds.append(cmd)
    return cmds


def _starvla_commands(episodes: int, seed: int) -> List[str]:
    """One command per suite for StarVLA via lerobot-eval (HF checkpoint, no repo)."""
    policy_path = "StarVLA/Qwen3-VL-PI-LIBERO-4in1"
    cmds = []
    for suite in LIBERO_4_SUITES:
        cmd = (
            f"lerobot-eval --policy.path={policy_path} --env.type=libero --env.task={suite} "
            f"--eval.n_episodes={episodes} --eval.seed={seed}"
        )
        cmds.append(cmd)
    return cmds


def get_commands(
    model_id: str,
    repo_path: Optional[str] = None,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    seed: int = DEFAULT_SEED,
    checkpoint: Optional[str] = None,
) -> List[str]:
    """Return list of shell commands to run LIBERO 4-suite eval for this model."""
    config = get_external_config(model_id)
    if not config:
        return []

    repo = repo_path or os.environ.get(f"{model_id.upper()}_REPO", "").strip() or "REPO_PATH"

    if model_id == "openvla":
        return _openvla_commands(repo, episodes_per_task, seed, checkpoint)
    if model_id == "ecot":
        return _openvla_commands(repo, episodes_per_task, seed, checkpoint or "Embodied-CoT/ecot-openvla-7b-bridge")
    if model_id == "xvla":
        return _xvla_commands(repo, episodes_per_task, seed)
    if model_id == "starvla":
        return _starvla_commands(episodes_per_task, seed)

    # Generic: use config template
    cmds = []
    ckpt = checkpoint or ""
    if config.command_per_suite:
        for suite in config.suites:
            cmds.append(
                config.command_per_suite.format(
                    repo_path=repo,
                    suite=suite,
                    episodes_per_task=episodes_per_task,
                    seed=seed,
                    checkpoint=ckpt,
                )
            )
    elif config.command_all_suites:
        cmds.append(
            config.command_all_suites.format(
                repo_path=repo,
                episodes_per_task=episodes_per_task,
                seed=seed,
                checkpoint=ckpt,
            )
        )
    return cmds


def run_commands(
    commands: List[str],
    print_only: bool = False,
    output_dir: Optional[str] = None,
    model_id: str = "external",
) -> int:
    """Run or print commands. If running, save combined stdout/stderr and a minimal results placeholder."""
    if print_only:
        for i, c in enumerate(commands):
            print(f"# Suite/run {i+1}")
            print(c)
            print()
        return 0

    output_dir = output_dir or os.path.join(_FRAMEWORK_ROOT, "eval", "results")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{model_id}_external_{ts}.log")
    results_path = os.path.join(output_dir, f"{model_id}_external_{ts}.json")

    all_out = []
    for i, cmd in enumerate(commands):
        print(f"Running ({i+1}/{len(commands)}): {cmd[:80]}...")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=_FRAMEWORK_ROOT,
            )
            out = result.stdout + "\n" + result.stderr
            all_out.append({"cmd": cmd, "returncode": result.returncode, "output": out})
            if result.returncode != 0:
                print(f"  [FAIL] returncode={result.returncode}")
            else:
                print(f"  [OK]")
        except subprocess.TimeoutExpired:
            all_out.append({"cmd": cmd, "returncode": -1, "output": "timeout"})
            print("  [TIMEOUT]")
        except Exception as e:
            all_out.append({"cmd": cmd, "returncode": -1, "output": str(e)})
            print(f"  [ERROR] {e}")

    with open(log_path, "w") as f:
        for item in all_out:
            f.write(f"CMD: {item['cmd']}\n")
            f.write(f"RETURNCODE: {item['returncode']}\n")
            f.write(item["output"])
            f.write("\n\n")
    print(f"Log written to {log_path}")

    # Placeholder result JSON (user can replace with parsed output)
    placeholder = {
        "model": model_id,
        "type": "external",
        "commands_run": [a["cmd"] for a in all_out],
        "returncodes": [a["returncode"] for a in all_out],
        "log_file": log_path,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(results_path, "w") as f:
        json.dump(placeholder, f, indent=2)
    print(f"Results placeholder written to {results_path}")
    return 0 if all(r["returncode"] == 0 for r in all_out) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external model LIBERO eval (4 suites).")
    parser.add_argument("--model", type=str, required=True, help="Model id: openvla, xvla, molmoact, deepthinkvla, ecot, internvla_m1, starvla, lightvla")
    parser.add_argument("--repo_path", type=str, default=None, help="Path to model repo (or set MODEL_REPO env var)")
    parser.add_argument("--episodes_per_task", type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint (model-dependent)")
    parser.add_argument("--print_cmd", action="store_true", help="Only print commands, do not run")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_id = args.model.lower().replace("-", "_")
    config = get_external_config(model_id)
    if not config:
        print(f"Unknown model: {args.model}. Known: openvla, xvla, molmoact, deepthinkvla, ecot, internvla_m1, starvla, lightvla", file=sys.stderr)
        return 1

    repo_path = args.repo_path or os.environ.get(f"{model_id.upper()}_REPO", "") or resolve_repo_path_from_dirs(model_id) or ""
    if not args.print_cmd and model_id not in ("xvla", "starvla") and not repo_path:
        print("Provide --repo_path, set REPO_PATH env, or put the repo in agent_attack_framework/repos/<model_id>.", file=sys.stderr)
        return 1

    commands = get_commands(
        model_id,
        repo_path=args.repo_path or (repo_path if repo_path and repo_path != "REPO_PATH" else None),
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    if not commands:
        print("No commands generated for this model. Check eval/external/configs.py.", file=sys.stderr)
        return 1

    return run_commands(
        commands,
        print_only=args.print_cmd,
        output_dir=args.output_dir,
        model_id=model_id,
    )


if __name__ == "__main__":
    sys.exit(main())
