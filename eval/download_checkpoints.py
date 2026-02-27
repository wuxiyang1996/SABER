#!/usr/bin/env python3
"""
Download Hugging Face checkpoints used for LIBERO evaluation.

Uses the repo_ids in eval/external/configs.HF_CHECKPOINTS_FOR_EVAL.
Downloads go to the Hugging Face cache (HF_HOME / HF_HUB_CACHE), so
evaluation tools (lerobot-eval, OpenVLA repo, etc.) will use them automatically.

Usage (from agent_attack_framework):
    pip install huggingface_hub
    python -m eval.download_checkpoints              # all models
    python -m eval.download_checkpoints --model xvla  # one model
    python -m eval.download_checkpoints --models openvla,starvla,xvla
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

# Prefer project cache for HF so downloads are in one place
_CACHE_ROOT = os.environ.get("AGENT_ATTACK_CACHE", os.path.join(_FRAMEWORK_ROOT, ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE_ROOT, "huggingface"))

from eval.external.configs import HF_CHECKPOINTS_FOR_EVAL


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face checkpoints for LIBERO evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Download checkpoints for this model only (e.g. xvla, starvla).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models (e.g. openvla,starvla,xvla). Overrides --model.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list repo_ids that would be downloaded; do not download.",
    )
    args = parser.parse_args()

    if args.models:
        model_ids = [m.strip().lower().replace("-", "_") for m in args.models.split(",")]
    elif args.model:
        model_ids = [args.model.strip().lower().replace("-", "_")]
    else:
        model_ids = list(HF_CHECKPOINTS_FOR_EVAL.keys())

    repo_ids_to_download: List[Tuple[str, str]] = []
    for mid in model_ids:
        if mid not in HF_CHECKPOINTS_FOR_EVAL:
            print(f"Unknown model: {mid}. Available: {list(HF_CHECKPOINTS_FOR_EVAL.keys())}", file=sys.stderr)
            return 1
        for repo_id in HF_CHECKPOINTS_FOR_EVAL[mid]:
            repo_ids_to_download.append((mid, repo_id))

    if args.list:
        for mid, repo_id in repo_ids_to_download:
            print(f"  {mid}: {repo_id}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"HF cache: {os.environ.get('HF_HUB_CACHE', 'default')}")
    print(f"Downloading {len(repo_ids_to_download)} checkpoint(s) ...")
    failed = []
    for mid, repo_id in repo_ids_to_download:
        print(f"  [{mid}] {repo_id} ...", end=" ", flush=True)
        try:
            path = snapshot_download(repo_id=repo_id)
            print(path)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((repo_id, str(e)))
    if failed:
        print(f"\n{len(failed)} download(s) failed:", file=sys.stderr)
        for repo_id, err in failed:
            print(f"  {repo_id}: {err}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
