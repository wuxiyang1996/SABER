"""ART training script for the adversarial attack agent.

Uses GRPO to train Qwen2.5-3B-Instruct to craft adversarial suffixes that
fool a frozen copy of the same model on HotpotQA questions.

The attack agent (LoRA-tuned) and the frozen eval model (base weights) are
both served from the same vLLM instance that ART's LocalBackend manages.

Usage:
    cd agent_attack_framework
    python -m trainer.train [--steps 50] [--lr 5e-6] [--train-samples 200]
"""

import argparse
import asyncio
import os
import random
import sys

from dotenv import load_dotenv
import weave

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import art
from art.local.backend import LocalBackend
from art.utils.iterate_dataset import iterate_dataset

from dataset.hotpotqa import load_hotpotqa
from agent.rollout import attack_rollout, set_frozen_eval_model
from art.langgraph import wrap_rollout
from eval_model.qa_model import FrozenQAModel

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ATTACK_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

EVAL_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_NAME = "hotpotqa-attack"
MODEL_NAME = "qwen2.5-3b-attacker"

DEFAULT_STEPS = 50
DEFAULT_LR = 5e-6
DEFAULT_TRAJECTORIES_PER_GROUP = 8
DEFAULT_GROUPS_PER_STEP = 4
DEFAULT_NUM_EPOCHS = 2
DEFAULT_EVAL_EVERY = 5
DEFAULT_TRAIN_SAMPLES = 200
DEFAULT_VAL_SAMPLES = 30


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

async def train(args: argparse.Namespace) -> None:
    weave.init(PROJECT_NAME)

    # ---- Backend & attack model (LoRA-tuned) ----
    backend = LocalBackend()

    attack_model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=ATTACK_BASE_MODEL,
    )

    print(f"Registering attack model ({ATTACK_BASE_MODEL}) ...")
    await attack_model.register(backend)

    # ---- Frozen eval model (Qwen2.5-3B-Instruct, separate from attack agent) ----
    # NOTE: The eval model uses a DIFFERENT base model than the attack agent.
    # It queries the vLLM server with the Qwen2.5-3B-Instruct model name.
    frozen_eval = FrozenQAModel(
        base_url=attack_model.inference_base_url,
        api_key=attack_model.inference_api_key,
        model_name=EVAL_BASE_MODEL,
    )
    await frozen_eval.resolve_model_name()
    set_frozen_eval_model(frozen_eval)
    print(f"Frozen eval model configured ({frozen_eval.model_name})")

    # ---- Dataset ----
    print("Loading HotpotQA dataset ...")
    train_scenarios = load_hotpotqa(
        split=args.split,
        max_samples=args.train_samples,
    )
    val_scenarios = load_hotpotqa(
        split="dev_distractor",
        max_samples=args.val_samples,
    )

    random.seed(42)

    # ---- Dataset iterator ----
    train_iter = iterate_dataset(
        train_scenarios,
        groups_per_step=args.groups_per_step,
        num_epochs=args.num_epochs,
        initial_step=await attack_model.get_step(),
    )

    # ---- Training loop ----
    print(f"\nStarting adversarial training for up to {args.steps} steps")
    print(f"  trajectories/group : {args.traj_per_group}")
    print(f"  groups/step        : {args.groups_per_step}")
    print(f"  learning rate      : {args.lr}")
    print()

    step_count = 0
    for batch in train_iter:
        if step_count >= args.steps:
            break

        print(f"\n{'='*60}")
        print(f"Step {batch.step}  (epoch {batch.epoch}, epoch_step {batch.epoch_step})")
        print(f"{'='*60}")

        # -- Gather attack trajectories --
        # wrap_rollout captures LLM interactions from the LangGraph ReAct
        # agent into the ART trajectory for GRPO training.
        wrapped_rollout = wrap_rollout(attack_model, attack_rollout)
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    wrapped_rollout(attack_model, scenario)
                    for _ in range(args.traj_per_group)
                )
                for scenario in batch.items
            ),
            pbar_desc=f"attack step {batch.step}",
        )

        # Print quick summary
        rewards = [t.reward for g in train_groups for t in g]
        successes = [t.metrics.get("attack_success", 0) for g in train_groups for t in g]
        if rewards:
            print(f"  mean reward       : {sum(rewards)/len(rewards):.3f}")
            print(f"  attack success    : {sum(successes)/len(successes):.1%}")

        # -- Train (GRPO) --
        result = await backend.train(attack_model, train_groups, learning_rate=args.lr)
        await attack_model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split="train",
        )

        # -- Periodic validation --
        if batch.step % args.eval_every == 0 and val_scenarios:
            print(f"\nValidation ({len(val_scenarios)} scenarios) ...")
            wrapped_val_rollout = wrap_rollout(attack_model, attack_rollout)
            val_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        wrapped_val_rollout(attack_model, scenario)
                        for _ in range(1)
                    )
                    for scenario in val_scenarios
                ),
                pbar_desc=f"val step {batch.step}",
            )
            await attack_model.log(val_groups, split="val", step=batch.step)

            val_rewards = [t.reward for g in val_groups for t in g]
            val_success = [t.metrics.get("attack_success", 0) for g in val_groups for t in g]
            val_f1 = [t.metrics.get("f1_adversarial", 0) for g in val_groups for t in g]
            if val_rewards:
                print(f"  val reward        : {sum(val_rewards)/len(val_rewards):.3f}")
                print(f"  val attack rate   : {sum(val_success)/len(val_success):.1%}")
                print(f"  val f1_adversarial: {sum(val_f1)/len(val_f1):.3f}")

        step_count += 1

    print("\nAdversarial training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train adversarial attack agent (ART + GRPO)"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--traj-per-group", type=int, default=DEFAULT_TRAJECTORIES_PER_GROUP)
    parser.add_argument("--groups-per-step", type=int, default=DEFAULT_GROUPS_PER_STEP)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--split", type=str, default="dev_distractor",
                        choices=["train", "dev_distractor", "dev_fullwiki"])
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
