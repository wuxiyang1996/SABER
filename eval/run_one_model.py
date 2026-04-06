#!/usr/bin/env python3
"""
Run a single VLA model on LIBERO (4 suites). Thin wrapper around run_libero_eval
that adds --gpu and forwards all other args.

Usage (from agent_attack_framework):
    python -m eval.run_one_model --model openpi_pi05
    python -m eval.run_one_model --model openvla --repo_path /path/to/openvla --gpu 0
    python -m eval.run_one_model --model ecot --gpu 1 --episodes_per_task 5 --seed 42
"""

from __future__ import annotations

import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))


def main() -> int:
    # Parse only --gpu here; forward everything else to run_libero_eval
    argv = sys.argv[1:]
    gpu = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1

    env = os.environ.copy()
    # Use same HF cache as eval.download_checkpoints so checkpoints load automatically
    _cache_root = os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache"))
    env.setdefault("HF_HOME", os.path.join(_cache_root, "huggingface"))
    env.setdefault("HF_HUB_CACHE", os.path.join(_cache_root, "huggingface", "hub"))
    env.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache_root, "huggingface"))
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    cmd = [sys.executable, "-m", "eval.run_libero_eval"] + rest
    return subprocess.run(cmd, cwd=_FRAMEWORK_ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
