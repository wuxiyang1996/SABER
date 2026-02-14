"""HotpotQA dataset loader and evaluation metrics.

Downloads the HotpotQA dataset and builds scenarios that include both the
question and the reference context (supporting paragraphs). The reference
context is what the frozen evaluation model sees alongside the question.
"""

import json
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class HotpotQAScenario:
    """A single HotpotQA question with its reference context.

    Attributes:
        question:       The original multi-hop question.
        answer:         Gold answer string.
        context:        Formatted reference paragraphs that the eval model sees.
        question_id:    Unique ID from the dataset.
        level:          Difficulty (easy / medium / hard).
        question_type:  Question type (bridge / comparison).
    """

    question: str
    answer: str
    context: str
    question_id: str
    level: str = "medium"
    question_type: str = "bridge"


# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

HOTPOTQA_DISTRACTOR_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
HOTPOTQA_FULLWIKI_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
)
HOTPOTQA_TRAIN_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
)


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------


def _build_context(item: dict) -> str:
    """Build a readable reference-context string from the HotpotQA item.

    The dataset provides a `context` field which is a list of
    [title, [sentence1, sentence2, ...]] pairs. We format them into titled
    paragraphs so the eval model gets clean input.
    """
    paragraphs: List[str] = []
    raw_context = item.get("context", [])

    for entry in raw_context:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        title = entry[0]
        sentences = entry[1]
        if isinstance(sentences, list):
            text = " ".join(str(s) for s in sentences)
        else:
            text = str(sentences)
        paragraphs.append(f"[{title}]\n{text}")

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_hotpotqa(
    split: str = "dev_distractor",
    max_samples: Optional[int] = None,
    cache_dir: str = ".cache/hotpotqa",
    difficulty: Optional[str] = None,
) -> List[HotpotQAScenario]:
    """Load HotpotQA scenarios with reference context.

    Args:
        split: 'train', 'dev_distractor', or 'dev_fullwiki'.
        max_samples: Cap on returned scenarios.
        cache_dir: Local cache directory.
        difficulty: Filter by 'easy', 'medium', or 'hard'.

    Returns:
        List of HotpotQAScenario objects.
    """
    url_map = {
        "train": HOTPOTQA_TRAIN_URL,
        "dev_distractor": HOTPOTQA_DISTRACTOR_URL,
        "dev_fullwiki": HOTPOTQA_FULLWIKI_URL,
    }
    if split not in url_map:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(url_map)}")

    url = url_map[split]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"hotpot_{split}.json")

    # Download if not cached
    if not os.path.exists(cache_path):
        print(f"Downloading HotpotQA {split} split ...")
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(resp.content)
        print(f"Saved to {cache_path}")

    with open(cache_path, "r") as f:
        raw = json.load(f)

    scenarios: List[HotpotQAScenario] = []
    for item in raw:
        level = item.get("level", "medium")
        if difficulty and level != difficulty:
            continue

        context = _build_context(item)
        if not context:
            continue  # skip items without usable context

        scenarios.append(
            HotpotQAScenario(
                question=item["question"],
                answer=item["answer"],
                context=context,
                question_id=item["_id"],
                level=level,
                question_type=item.get("type", "bridge"),
            )
        )

    if max_samples is not None:
        scenarios = scenarios[:max_samples]

    print(f"Loaded {len(scenarios)} HotpotQA scenarios (split={split})")
    return scenarios


# ---------------------------------------------------------------------------
# Evaluation metrics (standard HotpotQA F1 / EM)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction: str, gold: str) -> float:
    """Exact-match score (0 or 1)."""
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
