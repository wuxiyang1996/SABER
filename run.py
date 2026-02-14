#!/usr/bin/env python3
"""Convenience entry-point for the adversarial attack framework.

Can be invoked from any directory:
    python /home/wxy/agent_attack_framework/run.py train --steps 50 --lr 5e-6
    python /home/wxy/agent_attack_framework/run.py eval  --n 10
"""

import os
import sys

# Always resolve imports relative to this file's directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

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

    args = parser.parse_args()

    import asyncio

    if args.command == "train":
        from trainer.train import train
        asyncio.run(train(args))
    elif args.command == "eval":
        from eval import evaluate
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    cli()
