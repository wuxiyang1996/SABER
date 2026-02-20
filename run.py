#!/usr/bin/env python3
"""Convenience entry-point for the adversarial attack framework.

HotpotQA (text-only):
    python run.py train --steps 50 --lr 5e-6
    python run.py eval --n 10

Attack pi0.5 model in LIBERO (VLA):
    python run.py vla --objective task_failure --task_suite libero_spatial --task_ids 0,1,2
"""

import os
import sys

# Always resolve imports relative to this file's directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Cache under vlm-robot/.cache (before any imports that use HF/OpenPI/PyTorch caches)
_CACHE_ROOT = os.path.realpath(os.path.join(PROJECT_ROOT, "..", ".cache"))
os.environ.setdefault("OPENPI_DATA_HOME", _CACHE_ROOT)
os.environ.setdefault("HF_HOME", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE_ROOT, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_CACHE_ROOT, "torch"))
try:
    os.makedirs(_CACHE_ROOT, exist_ok=True)
except OSError:
    pass

import argparse


def cli():
    parser = argparse.ArgumentParser(
        description="HotpotQA Adversarial Attack Framework (ART + GRPO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train sub-command ----
    train_p = subparsers.add_parser("train", help="Train the attack agent with GRPO")
    train_p.add_argument("--steps", type=int, default=50)
    train_p.add_argument("--lr", type=float, default=5e-6)
    train_p.add_argument("--traj-per-group", type=int, default=8)
    train_p.add_argument("--groups-per-step", type=int, default=4)
    train_p.add_argument("--num-epochs", type=int, default=2)
    train_p.add_argument("--eval-every", type=int, default=5)
    train_p.add_argument("--split", type=str, default="dev_distractor",
                         choices=["train", "dev_distractor", "dev_fullwiki"])
    train_p.add_argument("--train-samples", type=int, default=200)
    train_p.add_argument("--val-samples", type=int, default=30)

    # ---- eval sub-command ----
    eval_p = subparsers.add_parser("eval", help="Evaluate attack agent vs baseline")
    eval_p.add_argument("--n", type=int, default=10)
    eval_p.add_argument("--attack-model", type=str, default="Qwen/Qwen3-1.7B")
    eval_p.add_argument("--eval-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    # ---- vla sub-command: attack pi0.5 model in LIBERO ----
    subparsers.add_parser(
        "vla",
        help="Train attack agent to perturb instructions/observations for pi0.5 in LIBERO (GRPO)",
    )

    args, unknown = parser.parse_known_args()

    import asyncio

    if args.command == "train":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        from trainer.train import train
        asyncio.run(train(args))
    elif args.command == "eval":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        from eval import evaluate
        asyncio.run(evaluate(args))
    elif args.command == "vla":
        # Pass through to train_vla (attack pi0.5 in LIBERO)
        sys.argv = [sys.argv[0]] + unknown
        from train_vla import main
        main()


if __name__ == "__main__":
    cli()
