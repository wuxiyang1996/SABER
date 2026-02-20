"""Post-training evaluation of the adversarial attack agent.

After training completes, this script measures how effectively the trained
attack agent can degrade the frozen eval model's QA performance on a
held-out set of HotpotQA questions.

It runs three passes:
  1. BASELINE  – eval model answers the original (clean) questions.
  2. ATTACK    – trained attack agent generates a suffix for each question,
                 then the eval model answers the modified question.
  3. REPORT    – side-by-side comparison of baseline vs attacked performance.

Usage:
    cd agent_attack_framework
    python eval_attack.py [--n 100] [--split dev_distractor]
"""

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import art
from art.local.backend import LocalBackend

from dataset.hotpotqa import load_hotpotqa, HotpotQAScenario, compute_f1, compute_em
from eval_model.qa_model import FrozenQAModel
from agent.rollout import attack_rollout, set_frozen_eval_model
from art.langgraph import wrap_rollout
from tools.attack import apply_suffix

load_dotenv()

# ---------------------------------------------------------------------------
# Model configs (must match trainer/train.py)
# ---------------------------------------------------------------------------

ATTACK_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EVAL_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_NAME = "hotpotqa-attack"
MODEL_NAME = "qwen2.5-3b-attacker"


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

class QuestionResult:
    """Stores baseline and attacked results for a single question."""

    def __init__(self, scenario: HotpotQAScenario):
        self.scenario = scenario
        # Baseline (clean question)
        self.baseline_answer: str = ""
        self.baseline_f1: float = 0.0
        self.baseline_em: float = 0.0
        # Attack (question + suffix / perturbed via any tool)
        self.suffix: str = ""
        self.adversarial_question: str = ""
        self.attack_answer: str = ""
        self.attack_f1: float = 0.0
        self.attack_em: float = 0.0
        self.tools_used: list[str] = []
        self.perturbation_length: int = 0
        self.attacked: bool = True  # explicit label: rollout had attack agent applied
        # Derived
        self.f1_drop: float = 0.0
        self.attack_success: bool = False  # baseline correct → attack incorrect


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

async def run_evaluation(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  HotpotQA Adversarial Attack — Post-Training Evaluation")
    print("=" * 70)

    # ---- Set up backend & models ----
    backend = LocalBackend()

    attack_model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=ATTACK_BASE_MODEL,
    )

    print(f"\nRegistering attack model ({ATTACK_BASE_MODEL}) ...")
    await attack_model.register(backend)

    current_step = await attack_model.get_step()
    print(f"  Loaded checkpoint at step {current_step}")

    frozen_eval = FrozenQAModel(
        base_url=attack_model.inference_base_url,
        api_key=attack_model.inference_api_key,
        model_name=EVAL_BASE_MODEL,
    )
    await frozen_eval.resolve_model_name()
    set_frozen_eval_model(frozen_eval)
    print(f"  Frozen eval model: {frozen_eval.model_name}")

    # ---- Load dataset ----
    scenarios = load_hotpotqa(
        split=args.split,
        max_samples=args.n,
    )
    print(f"  Evaluating on {len(scenarios)} questions (split={args.split})\n")

    results: list[QuestionResult] = []

    # ================================================================
    # PASS 1: Baseline — eval model on clean questions
    # ================================================================
    print("-" * 70)
    print("  PASS 1: Baseline (eval model on clean questions)")
    print("-" * 70)
    t0 = time.time()

    for i, scenario in enumerate(scenarios):
        answer = await frozen_eval.answer(scenario.question, scenario.context)
        r = QuestionResult(scenario)
        r.baseline_answer = answer
        r.baseline_f1 = compute_f1(answer, scenario.answer)
        r.baseline_em = compute_em(answer, scenario.answer)
        results.append(r)

        if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
            avg_f1 = sum(r.baseline_f1 for r in results) / len(results)
            print(f"  [{i+1}/{len(scenarios)}]  running baseline F1: {avg_f1:.3f}")

    baseline_time = time.time() - t0
    print(f"  Baseline pass done in {baseline_time:.1f}s\n")

    # ================================================================
    # PASS 2: Attack — agent generates suffix, eval model answers
    # ================================================================
    print("-" * 70)
    print("  PASS 2: Attack (attacked=True, trained agent generates adversarial suffixes)")
    print("-" * 70)
    t0 = time.time()

    wrapped_rollout = wrap_rollout(attack_model, attack_rollout)
    for i, (scenario, r) in enumerate(zip(scenarios, results)):
        traj = await wrapped_rollout(attack_model, scenario)

        # Rollout is explicitly labeled as attacked in trajectory metadata/metrics
        r.suffix = traj.metadata.get("suffix", "")
        r.adversarial_question = traj.metadata.get("adversarial_question", "")
        r.attack_answer = traj.metadata.get("eval_answer", "")
        r.attack_f1 = traj.metrics.get("f1_adversarial", 0.0)
        r.attack_em = traj.metrics.get("em_adversarial", 0.0)
        _tools_raw = traj.metadata.get("tools_used", "")
        r.tools_used = [t.strip() for t in _tools_raw.split(",") if t.strip()] if isinstance(_tools_raw, str) else list(_tools_raw or [])
        r.perturbation_length = traj.metrics.get("perturbation_length", 0)
        # Explicit attacked label from rollout (default True for attack pass)
        r.attacked = traj.metadata.get("attacked", True)

        # F1 drop = how much worse the eval model did after the attack
        r.f1_drop = r.baseline_f1 - r.attack_f1

        # Attack success = baseline was correct (EM=1) but attack made it wrong
        r.attack_success = (r.baseline_em == 1.0 and r.attack_em == 0.0)

        if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
            avg_atk_f1 = sum(r.attack_f1 for r in results if r.suffix is not None) / (i + 1)
            print(f"  [{i+1}/{len(scenarios)}]  running attack F1 (attacked=True): {avg_atk_f1:.3f}")

    attack_time = time.time() - t0
    print(f"  Attack pass done in {attack_time:.1f}s\n")

    # ================================================================
    # REPORT
    # ================================================================
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    n = len(results)

    # Aggregate baseline metrics
    avg_baseline_f1 = sum(r.baseline_f1 for r in results) / n
    avg_baseline_em = sum(r.baseline_em for r in results) / n

    # Aggregate attack metrics
    avg_attack_f1 = sum(r.attack_f1 for r in results) / n
    avg_attack_em = sum(r.attack_em for r in results) / n

    # F1 drop
    avg_f1_drop = sum(r.f1_drop for r in results) / n

    # Attack success rate (only among questions the baseline got right)
    baseline_correct = [r for r in results if r.baseline_em == 1.0]
    n_baseline_correct = len(baseline_correct)
    n_attack_success = sum(1 for r in baseline_correct if r.attack_success)
    attack_success_rate = n_attack_success / n_baseline_correct if n_baseline_correct > 0 else 0.0

    # Perturbation stats
    suffix_lengths = [len(r.suffix) for r in results if r.suffix]
    avg_suffix_len = sum(suffix_lengths) / len(suffix_lengths) if suffix_lengths else 0
    perturbation_lengths = [r.perturbation_length for r in results]
    avg_perturb_len = sum(perturbation_lengths) / n if perturbation_lengths else 0

    # Tool usage stats
    from collections import Counter
    tool_counter: Counter[str] = Counter()
    for r in results:
        tool_counter.update(r.tools_used)

    print(f"\n  Questions evaluated     : {n}")
    print(f"  Attack model checkpoint : step {current_step}")
    print()
    print(f"  {'Metric':<28} {'Baseline':>10} {'Attacked':>10} {'Delta':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'F1 (avg)':<28} {avg_baseline_f1:>10.3f} {avg_attack_f1:>10.3f} {avg_attack_f1 - avg_baseline_f1:>+10.3f}")
    print(f"  {'EM (avg)':<28} {avg_baseline_em:>10.3f} {avg_attack_em:>10.3f} {avg_attack_em - avg_baseline_em:>+10.3f}")
    print()
    print(f"  Baseline EM accuracy    : {avg_baseline_em:.1%}  ({n_baseline_correct}/{n} correct)")
    print(f"  Attack success rate     : {attack_success_rate:.1%}  ({n_attack_success}/{n_baseline_correct} flipped)")
    print(f"  Avg F1 drop             : {avg_f1_drop:+.3f}")
    print(f"  Avg suffix length       : {avg_suffix_len:.0f} chars")
    print(f"  Avg perturbation length : {avg_perturb_len:.0f} chars")
    if tool_counter:
        print(f"\n  Tool usage (across all questions):")
        for tool_name, count in tool_counter.most_common():
            print(f"    {tool_name:<30} : {count}")
    print()

    # ---- Per-question details (top successes & failures) ----
    # Sort by F1 drop (biggest drops = most successful attacks)
    sorted_by_drop = sorted(results, key=lambda r: r.f1_drop, reverse=True)

    print("-" * 70)
    print("  TOP 10 MOST SUCCESSFUL ATTACKS (largest F1 drop)")
    print("-" * 70)
    for rank, r in enumerate(sorted_by_drop[:10], 1):
        print(f"\n  #{rank}  F1: {r.baseline_f1:.2f} → {r.attack_f1:.2f}  (drop={r.f1_drop:+.2f})")
        print(f"    Q : {r.scenario.question[:100]}")
        print(f"    Gold   : {r.scenario.answer}")
        print(f"    Base   : {r.baseline_answer}")
        print(f"    Attack : {r.attack_answer}")
        suffix_preview = r.suffix[:120] + ("..." if len(r.suffix) > 120 else "")
        print(f"    Suffix : {suffix_preview}")
        if r.tools_used:
            print(f"    Tools  : {', '.join(r.tools_used)}")

    # Questions where attack failed (eval model resisted)
    resisted = [r for r in results if r.f1_drop <= 0 and r.baseline_em == 1.0]
    if resisted:
        print(f"\n{'-'*70}")
        print(f"  EVAL MODEL RESISTED ATTACK: {len(resisted)} questions (baseline EM=1, no F1 drop)")
        print(f"{'-'*70}")
        for r in resisted[:5]:
            print(f"    Q: {r.scenario.question[:90]}")
            print(f"    Gold: {r.scenario.answer}  |  Attack answer: {r.attack_answer}")
            print(f"    Suffix: {r.suffix[:80]}")
            print()

    # ---- Save detailed results to JSON ----
    output_path = os.path.join(
        ".art", PROJECT_NAME, "eval_results",
        f"eval_step{current_step}_{args.split}_n{n}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    json_results = {
        "config": {
            "attack_model": ATTACK_BASE_MODEL,
            "eval_model": EVAL_BASE_MODEL,
            "checkpoint_step": current_step,
            "split": args.split,
            "n_questions": n,
            "rollout_type": "attacked",
            "attacked": True,
        },
        "summary": {
            "baseline_f1": avg_baseline_f1,
            "baseline_em": avg_baseline_em,
            "attack_f1": avg_attack_f1,
            "attack_em": avg_attack_em,
            "f1_drop": avg_f1_drop,
            "attack_success_rate": attack_success_rate,
            "n_baseline_correct": n_baseline_correct,
            "n_attack_success": n_attack_success,
            "avg_suffix_length": avg_suffix_len,
        },
        "per_question": [
            {
                "question_id": r.scenario.question_id,
                "question": r.scenario.question,
                "gold_answer": r.scenario.answer,
                "level": r.scenario.level,
                "type": r.scenario.question_type,
                "baseline_answer": r.baseline_answer,
                "baseline_f1": r.baseline_f1,
                "baseline_em": r.baseline_em,
                "attacked": r.attacked,
                "rollout_type": "attacked" if r.attacked else "baseline",
                "suffix": r.suffix,
                "attack_answer": r.attack_answer,
                "attack_f1": r.attack_f1,
                "attack_em": r.attack_em,
                "f1_drop": r.f1_drop,
                "attack_success": r.attack_success,
                "tools_used": r.tools_used,
                "perturbation_length": r.perturbation_length,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained attack agent vs frozen eval model on HotpotQA"
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of questions to evaluate")
    parser.add_argument("--split", type=str, default="dev_distractor",
                        choices=["train", "dev_distractor", "dev_fullwiki"],
                        help="HotpotQA split to evaluate on")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
