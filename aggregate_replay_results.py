#!/usr/bin/env python3
"""Aggregate per-model replay evaluation results into a cross-model report.

Reads all ``replay_*.json`` files from an input directory and produces:
  1. A cross-model comparison table (printed and saved as JSON)
  2. Per-suite breakdown across models
  3. Source-victim comparison (which source's attacks transfer better)

Usage:
    python aggregate_replay_results.py \\
        --input_dir outputs/replay_eval_task_failure \\
        --output outputs/replay_eval_task_failure/cross_model_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("aggregate_replay")


def load_replay_reports(input_dir: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(input_dir, "replay_*.json")
    files = sorted(glob.glob(pattern))
    reports = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
            data["_source_file"] = os.path.basename(f)
            reports.append(data)
    return reports


def compute_per_suite_metrics(episodes: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Group episodes by suite and compute per-suite ASR and TER."""
    suite_data = defaultdict(list)
    for ep in episodes:
        suite_data[ep["task_suite"]].append(ep)

    result = {}
    for suite, eps in sorted(suite_data.items()):
        n = len(eps)
        bl_success = sum(1 for e in eps if e["baseline"]["success"])
        atk_success = sum(1 for e in eps if e["attack"]["success"])
        flipped = sum(1 for e in eps if e.get("flipped", False))
        bl_steps = [e["baseline"]["steps"] for e in eps]
        atk_steps = [e["attack"]["steps"] for e in eps]

        result[suite] = {
            "num_episodes": n,
            "baseline_ter": bl_success / n if n else 0.0,
            "attack_ter": atk_success / n if n else 0.0,
            "asr": flipped / bl_success if bl_success else 0.0,
            "num_flipped": flipped,
            "num_baseline_success": bl_success,
            "avg_baseline_steps": float(np.mean(bl_steps)) if bl_steps else 0.0,
            "avg_attack_steps": float(np.mean(atk_steps)) if atk_steps else 0.0,
        }
    return result


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build cross-model aggregation from replay reports."""

    per_model = {}
    per_source = defaultdict(list)

    for report in reports:
        config = report["config"]
        victim = config["victim_model"]
        source = config["source_victim"]
        key = f"{victim}_from_{source}"

        bl = report["baseline_summary"]
        atk = report["attack_summary"]
        episodes = report.get("per_episode", [])

        suite_metrics = compute_per_suite_metrics(episodes)

        entry = {
            "victim_model": victim,
            "source_victim": source,
            "num_episodes": bl["num_episodes"],
            "baseline_ter": bl["task_execution_rate"],
            "attack_ter": atk["task_execution_rate"],
            "ter_delta": atk["task_execution_rate"] - bl["task_execution_rate"],
            "asr": atk.get("attack_success_rate", 0.0),
            "num_flipped": atk.get("num_flipped", 0),
            "num_baseline_success": atk.get("num_baseline_success", 0),
            "avg_chars_changed": atk.get("avg_chars_changed", 0.0),
            "avg_baseline_steps": bl.get("avg_action_seq_length", 0.0),
            "avg_attack_steps": atk.get("avg_action_seq_length", 0.0),
            "step_ratio": atk.get("avg_step_ratio", 0.0),
            "per_suite": suite_metrics,
            "source_file": report.get("_source_file", ""),
        }
        per_model[key] = entry
        per_source[source].append(entry)

    # Cross-source comparison: for each source, what's the average ASR?
    source_summary = {}
    for source, entries in per_source.items():
        asrs = [e["asr"] for e in entries if e["num_baseline_success"] > 0]
        ter_deltas = [e["ter_delta"] for e in entries]
        source_summary[source] = {
            "num_victims": len(entries),
            "avg_asr": float(np.mean(asrs)) if asrs else 0.0,
            "std_asr": float(np.std(asrs)) if asrs else 0.0,
            "avg_ter_delta": float(np.mean(ter_deltas)),
            "victims": [e["victim_model"] for e in entries],
        }

    # Model-level summary: for each victim, average across sources
    victim_models = sorted(set(e["victim_model"] for e in per_model.values()))
    model_summary = {}
    for victim in victim_models:
        entries = [e for e in per_model.values() if e["victim_model"] == victim]
        asrs = [e["asr"] for e in entries if e["num_baseline_success"] > 0]
        ters = [e["baseline_ter"] for e in entries]
        model_summary[victim] = {
            "num_sources": len(entries),
            "avg_baseline_ter": float(np.mean(ters)) if ters else 0.0,
            "avg_asr": float(np.mean(asrs)) if asrs else 0.0,
            "std_asr": float(np.std(asrs)) if asrs else 0.0,
            "per_source": {
                e["source_victim"]: {
                    "asr": e["asr"],
                    "ter_delta": e["ter_delta"],
                    "num_flipped": e["num_flipped"],
                    "num_baseline_success": e["num_baseline_success"],
                }
                for e in entries
            },
        }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_reports": len(reports),
        "per_model_source": per_model,
        "model_summary": model_summary,
        "source_summary": source_summary,
    }


def print_summary(summary: Dict[str, Any]):
    """Print a formatted cross-model comparison table."""
    lines = [
        "",
        "=" * 90,
        "  CROSS-MODEL ATTACK TRANSFERABILITY SUMMARY (task_failure)",
        "=" * 90,
    ]

    # Per-victim summary
    lines.append("")
    lines.append(
        f"  {'Victim':<18} {'BL-TER':>7} {'Sources':>8} {'Avg ASR':>8} {'Std ASR':>8}"
    )
    lines.append("  " + "─" * 54)
    for victim, ms in sorted(summary["model_summary"].items()):
        lines.append(
            f"  {victim:<18} {ms['avg_baseline_ter']:>6.1%} "
            f"{ms['num_sources']:>8} "
            f"{ms['avg_asr']:>7.1%} "
            f"{ms['std_asr']:>7.3f}"
        )

    # Detailed per-model-per-source
    lines.append("")
    lines.append("  " + "─" * 86)
    lines.append(
        f"  {'Victim':<16} {'Source':<14} {'Episodes':>8} "
        f"{'BL-TER':>7} {'ATK-TER':>8} {'ASR':>6} "
        f"{'Flipped':>8} {'Chars':>6}"
    )
    lines.append("  " + "─" * 86)

    for key in sorted(summary["per_model_source"].keys()):
        e = summary["per_model_source"][key]
        lines.append(
            f"  {e['victim_model']:<16} {e['source_victim']:<14} "
            f"{e['num_episodes']:>8} "
            f"{e['baseline_ter']:>6.1%} "
            f"{e['attack_ter']:>7.1%} "
            f"{e['asr']:>5.1%} "
            f"{e['num_flipped']:>4}/{e['num_baseline_success']:<3} "
            f"{e['avg_chars_changed']:>5.1f}"
        )

    # Source comparison
    lines.append("")
    lines.append("  " + "─" * 60)
    lines.append("  Source attack transferability:")
    for source, ss in sorted(summary["source_summary"].items()):
        lines.append(
            f"    {source:<14}: avg ASR {ss['avg_asr']:.1%} "
            f"(±{ss['std_asr']:.3f}) across {ss['num_victims']} victims"
        )

    # Per-suite breakdown for each victim-source pair
    lines.append("")
    lines.append("  " + "─" * 86)
    lines.append("  Per-suite ASR breakdown:")
    for key in sorted(summary["per_model_source"].keys()):
        e = summary["per_model_source"][key]
        suites = e.get("per_suite", {})
        if suites:
            lines.append(f"    {e['victim_model']} ← {e['source_victim']}:")
            for suite, sm in sorted(suites.items()):
                lines.append(
                    f"      {suite:<16}: "
                    f"BL-TER={sm['baseline_ter']:.0%}  "
                    f"ATK-TER={sm['attack_ter']:.0%}  "
                    f"ASR={sm['asr']:.0%} "
                    f"({sm['num_flipped']}/{sm['num_baseline_success']})"
                )

    lines.append("=" * 90)
    text = "\n".join(lines)
    print(text)
    logger.info(text)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate replay evaluation results across models.",
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    reports = load_replay_reports(args.input_dir)
    if not reports:
        logger.warning("No replay_*.json files found in %s", args.input_dir)
        sys.exit(0)

    logger.info("Found %d replay report(s) in %s", len(reports), args.input_dir)

    summary = aggregate(reports)
    print_summary(summary)

    output = args.output or os.path.join(args.input_dir, "cross_model_summary.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", output)


if __name__ == "__main__":
    main()
