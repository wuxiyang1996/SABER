#!/usr/bin/env python3
"""
Compute LIBERO metrics per task category (libero_spatial, libero_object, libero_goal, libero_10)
from eval_result replay JSONs and agent_output_records_* JSONs.

Metrics (per paper):
  (i)   Average number of tool calls per episode by the attack agent
  (ii)  Character-level instruction edit distance (avg chars modified)
  (iii) Action-sequence length change (before vs after perturbation)
  (iv)  Task execution success rate (after attack)
  (v)   Average number of constraint violations per episode

Also reports: initial (baseline) task execution success rate per model per task category.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

# LIBERO task categories (suites)
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
OUTPUTS = FRAMEWORK_ROOT / "outputs"


def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return None


def infer_objective_and_victim_from_path(path: Path, data: dict) -> tuple[str, str]:
    """Infer objective (action_inflation, task_failure, constraint_violation) and victim model name."""
    name = path.stem
    victim = (data.get("config") or {}).get("victim_model", "unknown")
    if "action_inflation" in name or "action_inflation" in (data.get("config") or {}).get("source_objective", ""):
        return "action_inflation", victim
    if "constraint_violation" in name or "constraint_violation" in (data.get("config") or {}).get("source_objective", ""):
        return "constraint_violation", victim
    if "task_failure" in name:
        return "task_failure", victim
    return "unknown", victim


def aggregate_from_per_episode(per_episode: list) -> dict[str, dict]:
    """Aggregate metrics per task category from per_episode list."""
    by_suite = defaultdict(lambda: {
        "tool_calls": [],
        "chars_changed": [],
        "baseline_steps": [],
        "attack_steps": [],
        "baseline_success": [],
        "attack_success": [],
        "baseline_constraint_violations": [],
        "attack_constraint_violations": [],
    })
    for ep in per_episode:
        suite = ep.get("task_suite")
        if suite not in LIBERO_SUITES:
            continue
        by_suite[suite]["tool_calls"].append(ep.get("num_tool_calls", 0))
        by_suite[suite]["chars_changed"].append(ep.get("chars_changed", 0))
        base = ep.get("baseline") or {}
        att = ep.get("attack") or {}
        by_suite[suite]["baseline_steps"].append(base.get("steps", 0))
        by_suite[suite]["attack_steps"].append(att.get("steps", 0))
        by_suite[suite]["baseline_success"].append(1 if base.get("success") else 0)
        by_suite[suite]["attack_success"].append(1 if att.get("success") else 0)
        by_suite[suite]["baseline_constraint_violations"].append(base.get("constraint_violations", 0) or 0)
        by_suite[suite]["attack_constraint_violations"].append(att.get("constraint_violations", 0) or 0)

    result = {}
    for suite in LIBERO_SUITES:
        if suite not in by_suite or not by_suite[suite]["tool_calls"]:
            continue
        d = by_suite[suite]
        n = len(d["tool_calls"])
        result[suite] = {
            "num_episodes": n,
            "avg_tool_calls": sum(d["tool_calls"]) / n,
            "avg_char_edit_dist": sum(d["chars_changed"]) / n,
            "avg_baseline_steps": sum(d["baseline_steps"]) / n,
            "avg_attack_steps": sum(d["attack_steps"]) / n,
            "action_seq_length_change": (sum(d["attack_steps"]) - sum(d["baseline_steps"])) / n,
            "baseline_task_success_rate": sum(d["baseline_success"]) / n,
            "task_execution_success_rate": sum(d["attack_success"]) / n,
            "avg_constraint_violations": sum(d["attack_constraint_violations"]) / n,
        }
    return result


def aggregate_from_per_task(per_task: dict, baseline_summary: dict, attack_summary: dict) -> dict[str, dict]:
    """Fallback: aggregate from per_task keys (suite/task_id) when per_episode is missing."""
    by_suite = defaultdict(lambda: {
        "count": 0,
        "tool_calls_sum": 0,
        "chars_sum": 0,
        "baseline_steps_sum": 0,
        "attack_steps_sum": 0,
        "baseline_success_sum": 0,
        "attack_success_sum": 0,
        "constraint_viol_sum": 0,
    })
    for key, pt in per_task.items():
        if "/" not in key:
            continue
        suite = key.split("/")[0]
        if suite not in LIBERO_SUITES:
            continue
        c = pt.get("count", 1)
        by_suite[suite]["count"] += c
        by_suite[suite]["tool_calls_sum"] += pt.get("avg_tools_called", 0) * c
        by_suite[suite]["chars_sum"] += pt.get("avg_chars_changed", 0) * c
        by_suite[suite]["attack_steps_sum"] += pt.get("avg_action_seq_length", 0) * c
        by_suite[suite]["attack_success_sum"] += pt.get("task_execution_rate", 0) * c
        by_suite[suite]["constraint_viol_sum"] += pt.get("avg_constraint_violations", 0) * c
        # baseline from baseline_success_rate if present
        base_sr = pt.get("baseline_success_rate")
        if base_sr is not None:
            by_suite[suite]["baseline_success_sum"] += base_sr * c
        base_steps = pt.get("avg_baseline_steps")
        if base_steps is not None:
            by_suite[suite]["baseline_steps_sum"] += base_steps * c

    result = {}
    for suite in LIBERO_SUITES:
        if suite not in by_suite or by_suite[suite]["count"] == 0:
            continue
        d = by_suite[suite]
        n = d["count"]
        baseline_success_rate = d["baseline_success_sum"] / n if d["baseline_success_sum"] or "baseline_success_sum" in str(d) else None
        avg_baseline_steps = d["baseline_steps_sum"] / n if d["baseline_steps_sum"] else None
        avg_attack_steps = d["attack_steps_sum"] / n
        result[suite] = {
            "num_episodes": n,
            "avg_tool_calls": d["tool_calls_sum"] / n,
            "avg_char_edit_dist": d["chars_sum"] / n,
            "avg_baseline_steps": avg_baseline_steps,
            "avg_attack_steps": avg_attack_steps,
            "action_seq_length_change": (avg_attack_steps - avg_baseline_steps) if avg_baseline_steps is not None else None,
            "baseline_task_success_rate": baseline_success_rate,
            "task_execution_success_rate": d["attack_success_sum"] / n,
            "avg_constraint_violations": d["constraint_viol_sum"] / n,
        }
    return result


def process_file(path: Path, data: dict) -> dict | None:
    """Compute per-suite metrics for one result file. Returns dict with per_suite and averaged."""
    per_episode = data.get("per_episode", [])
    per_task = data.get("per_task", {})
    baseline_summary = data.get("baseline_summary", {})
    attack_summary = data.get("attack_summary", {})

    if per_episode:
        per_suite = aggregate_from_per_episode(per_episode)
    elif per_task:
        per_suite = aggregate_from_per_task(per_task, baseline_summary, attack_summary)
    else:
        return None

    if not per_suite:
        return None

    # Averaged over suites (for this file)
    n_total = sum(s["num_episodes"] for s in per_suite.values())
    avg_tool_calls = sum(s["avg_tool_calls"] * s["num_episodes"] for s in per_suite.values()) / n_total
    avg_char_edit = sum(s["avg_char_edit_dist"] * s["num_episodes"] for s in per_suite.values()) / n_total
    avg_action_delta = None
    if all(s.get("action_seq_length_change") is not None for s in per_suite.values()):
        avg_action_delta = sum(s["action_seq_length_change"] * s["num_episodes"] for s in per_suite.values()) / n_total
    task_success = sum(s["task_execution_success_rate"] * s["num_episodes"] for s in per_suite.values()) / n_total
    avg_cv = sum(s["avg_constraint_violations"] * s["num_episodes"] for s in per_suite.values()) / n_total
    baseline_success = None
    if all(s.get("baseline_task_success_rate") is not None for s in per_suite.values()):
        baseline_success = sum(s["baseline_task_success_rate"] * s["num_episodes"] for s in per_suite.values()) / n_total

    objective, victim = infer_objective_and_victim_from_path(path, data)
    return {
        "source_path": str(path.relative_to(FRAMEWORK_ROOT) if FRAMEWORK_ROOT in path.parents else path),
        "victim_model": victim,
        "objective": objective,
        "per_suite": per_suite,
        "averaged": {
            "num_episodes": n_total,
            "baseline_task_success_rate": baseline_success,
            "avg_tool_calls": avg_tool_calls,
            "avg_char_edit_dist": avg_char_edit,
            "action_seq_length_change": avg_action_delta,
            "task_execution_success_rate": task_success,
            "avg_constraint_violations": avg_cv,
        },
    }


def collect_result_paths() -> list[Path]:
    paths = []
    # eval_result: all replay_*.json
    eval_result = OUTPUTS / "eval_result"
    if eval_result.exists():
        for f in eval_result.glob("replay_*.json"):
            paths.append(f)
    # agent_output_records_*
    for folder in ["agent_output_records_action_inflation", "agent_output_records_task_failure_2", "agent_output_records_constraint_violation"]:
        folder_path = OUTPUTS / folder
        if folder_path.exists():
            for f in folder_path.glob("*.json"):
                paths.append(f)
    return sorted(paths)


def main():
    paths = collect_result_paths()
    print(f"Found {len(paths)} result JSON files.\n")

    results = []
    for path in paths:
        data = load_json(path)
        if not data:
            continue
        out = process_file(path, data)
        if out:
            results.append(out)

    # --- Text report ---
    print("=" * 100)
    print("LIBERO METRICS BY TASK CATEGORY (and averaged)")
    print("(i) avg tool calls/episode  (ii) char edit dist  (iii) action seq length change  (iv) task success rate  (v) avg constraint violations/episode")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r['victim_model']} | {r['objective']} | {r['source_path']} ---")
        print("Per task category:")
        for suite in LIBERO_SUITES:
            if suite not in r["per_suite"]:
                continue
            s = r["per_suite"][suite]
            base_sr = s.get("baseline_task_success_rate")
            base_sr_str = f"{base_sr:.3f}" if base_sr is not None else "N/A"
            delta_str = f"{s['action_seq_length_change']:.1f}" if s.get("action_seq_length_change") is not None else "N/A"
            print(f"  {suite}: baseline_ter={base_sr_str}  tool_calls={s['avg_tool_calls']:.2f}  char_edit={s['avg_char_edit_dist']:.1f}  "
                  f"action_delta={delta_str}  task_success={s['task_execution_success_rate']:.3f}  constraint_viol={s['avg_constraint_violations']:.1f}  (n={s['num_episodes']})")
        a = r["averaged"]
        base_avg = a.get("baseline_task_success_rate")
        base_avg_str = f"{base_avg:.3f}" if base_avg is not None else "N/A"
        delta_avg_str = f"{a['action_seq_length_change']:.1f}" if a.get("action_seq_length_change") is not None else "N/A"
        print("Averaged (all categories):")
        print(f"  baseline_ter={base_avg_str}  avg_tool_calls={a['avg_tool_calls']:.2f}  avg_char_edit={a['avg_char_edit_dist']:.1f}  "
              f"action_seq_length_change={delta_avg_str}  task_success_rate={a['task_execution_success_rate']:.3f}  avg_constraint_violations={a['avg_constraint_violations']:.1f}  (n={a['num_episodes']})")

    # --- Summary table: averaged metrics only (for paper) ---
    print("\n" + "=" * 100)
    print("SUMMARY: AVERAGED METRICS (all LIBERO categories) + BASELINE TASK SUCCESS RATE")
    print("=" * 100)
    print(f"{'Victim Model':<22} {'Objective':<22} {'Baseline TER':<14} {'Tool Calls':<12} {'Char Edit':<10} {'Action Δ':<10} {'Task Success':<14} {'Const. Viol.':<12}")
    print("-" * 100)
    for r in results:
        a = r["averaged"]
        base = a.get("baseline_task_success_rate")
        base_str = f"{base:.3f}" if base is not None else "N/A"
        delta = a.get("action_seq_length_change")
        delta_str = f"{delta:.1f}" if delta is not None else "N/A"
        print(f"{r['victim_model']:<22} {r['objective']:<22} {base_str:<14} {a['avg_tool_calls']:<12.2f} {a['avg_char_edit_dist']:<10.1f} {delta_str:<10} {a['task_execution_success_rate']:<14.3f} {a['avg_constraint_violations']:<12.1f}")

    # --- JSON output ---
    out_path = FRAMEWORK_ROOT / "outputs" / "libero_metrics_by_category.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize for JSON (no lambdas)
    json_out = []
    for r in results:
        json_out.append({
            "source_path": r["source_path"],
            "victim_model": r["victim_model"],
            "objective": r["objective"],
            "per_suite": r["per_suite"],
            "averaged": r["averaged"],
        })
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nFull per-category data written to: {out_path}")

    # --- Markdown report (averaged + per-category baseline TER) ---
    md_path = FRAMEWORK_ROOT / "outputs" / "libero_metrics_report.md"
    with open(md_path, "w") as f:
        f.write("# LIBERO metrics by task category\n\n")
        f.write("**Metrics.** (i) Avg tool calls/episode, (ii) char-level instruction edit distance, ")
        f.write("(iii) action-sequence length change (after − before), (iv) task execution success rate, ")
        f.write("(v) avg constraint violations/episode. Plus **baseline (initial) task execution success rate** per model per task.\n\n")
        f.write("## Averaged metrics (over all LIBERO categories)\n\n")
        f.write("| Victim Model | Objective | Baseline TER | Tool Calls | Char Edit | Action Δ | Task Success | Const. Viol. |\n")
        f.write("|--------------|-----------|-------------|------------|-----------|----------|--------------|-------------|\n")
        for r in results:
            a = r["averaged"]
            base = a.get("baseline_task_success_rate")
            base_str = f"{base:.3f}" if base is not None else "—"
            delta = a.get("action_seq_length_change")
            delta_str = f"{delta:.1f}" if delta is not None else "—"
            f.write(f"| {r['victim_model']} | {r['objective']} | {base_str} | {a['avg_tool_calls']:.2f} | {a['avg_char_edit_dist']:.1f} | {delta_str} | {a['task_execution_success_rate']:.3f} | {a['avg_constraint_violations']:.1f} |\n")
        f.write("\n## Per-task-category: baseline TER and metrics\n\n")
        for r in results:
            f.write(f"### {r['victim_model']} — {r['objective']}\n\n")
            f.write("| Category | Baseline TER | Tool Calls | Char Edit | Action Δ | Task Success | Const. Viol. | n |\n")
            f.write("|----------|--------------|------------|-----------|----------|--------------|-------------|---|\n")
            for suite in LIBERO_SUITES:
                if suite not in r["per_suite"]:
                    continue
                s = r["per_suite"][suite]
                base_sr = s.get("baseline_task_success_rate")
                base_str = f"{base_sr:.3f}" if base_sr is not None else "—"
                delta = s.get("action_seq_length_change")
                delta_str = f"{delta:.1f}" if delta is not None else "—"
                f.write(f"| {suite} | {base_str} | {s['avg_tool_calls']:.2f} | {s['avg_char_edit_dist']:.1f} | {delta_str} | {s['task_execution_success_rate']:.3f} | {s['avg_constraint_violations']:.1f} | {s['num_episodes']} |\n")
            f.write("\n")
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
