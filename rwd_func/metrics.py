"""Evaluation metrics for adversarial VLA attack experiments.

Computes aggregate metrics across a batch of trajectories (from ART
trajectory groups) produced by the VLA attack training pipeline.

Paper metrics
-------------
1. **Avg Tools Called** — mean number of attack tools invoked per episode.
2. **Avg Chars Changed** — mean character-level edit distance between the
   original and perturbed instruction.
3. **Action Token Sequence Length** — mean VLA action steps per episode.
4. **Task Execution Rate** — fraction where VLA still succeeded under attack.
5. **Constraint Violations** — mean total violation count per episode.

Additional attack-performance metrics
--------------------------------------
6. **Attack Success Rate (ASR)** — fraction of episodes where the attack
   flipped a baseline-successful task to failure.
7. **Stealth / perturbation** — token edit ratio, char edit distance,
   added tokens, visual L∞, pixel change ratio, SSIM.
8. **Perturbation type distribution** — breakdown of which tool families
   (token, char, prompt, visual) were used.
9. **Baseline vs attack step ratio** — how many more steps the VLA takes
   under attack compared to the baseline.
10. **Action divergence** — mean L2 distance between baseline and attack
    action trajectories (when same length) to quantify behavioural shift.
11. **Per-task breakdown** — metrics grouped by task suite and task ID.

Usage
-----
::

    from rwd_func.metrics import compute_metrics, print_metrics

    metrics = compute_metrics(groups)           # from ART trajectory groups
    print_metrics(metrics, step=step_num)       # formatted table
    metrics.to_dict()                           # flat dict for W&B / CSV
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rwd_func.rwd import edit_distance, token_edit_ratio

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class AttackMetrics:
    """Aggregated evaluation metrics for a batch of attack episodes."""

    num_episodes: int = 0

    # --- 1. Tools called ---
    avg_tools_called: float = 0.0
    std_tools_called: float = 0.0

    # --- 2. Characters changed (char-level edit distance) ---
    avg_chars_changed: float = 0.0
    std_chars_changed: float = 0.0

    # --- 3. Action token sequence length (VLA env steps under attack) ---
    avg_action_seq_length: float = 0.0
    std_action_seq_length: float = 0.0

    # --- 4. Task execution rate (VLA success under attack) ---
    task_execution_rate: float = 0.0

    # --- 5. Constraint violations ---
    avg_constraint_violations: float = 0.0
    std_constraint_violations: float = 0.0
    avg_collisions: float = 0.0
    avg_joint_limit_violations: float = 0.0
    avg_excessive_force: float = 0.0

    # --- 6. Attack success rate (ASR) ---
    attack_success_rate: float = 0.0
    num_baseline_success: int = 0
    num_flipped: int = 0

    # --- 7. Stealth / perturbation magnitude ---
    avg_token_edit_ratio: float = 0.0
    avg_char_edit_dist: float = 0.0
    avg_stealth_penalty: float = 0.0
    avg_visual_linf: float = 0.0
    avg_visual_pixel_change: float = 0.0
    avg_visual_ssim: float = 0.0

    # --- 8. Perturbation type distribution ---
    tool_family_counts: Dict[str, int] = field(default_factory=dict)
    tool_family_fractions: Dict[str, float] = field(default_factory=dict)

    # --- 9. Step ratio (attack steps / baseline steps) ---
    avg_step_ratio: float = 0.0
    avg_baseline_steps: float = 0.0

    # --- 10. Action divergence ---
    avg_action_divergence: float = 0.0

    # --- General ---
    avg_reward: float = 0.0
    std_reward: float = 0.0
    baseline_success_rate: float = 0.0

    # --- 11. Per-task breakdown ---
    per_task: Dict[str, "AttackMetricsSummary"] = field(default_factory=dict)

    # Per-episode raw values (for histograms / statistical tests)
    per_episode: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Flat dict suitable for logging / W&B."""
        d: Dict[str, Any] = {
            "num_episodes": self.num_episodes,
            "avg_tools_called": self.avg_tools_called,
            "std_tools_called": self.std_tools_called,
            "avg_chars_changed": self.avg_chars_changed,
            "std_chars_changed": self.std_chars_changed,
            "avg_action_seq_length": self.avg_action_seq_length,
            "std_action_seq_length": self.std_action_seq_length,
            "task_execution_rate": self.task_execution_rate,
            "avg_constraint_violations": self.avg_constraint_violations,
            "std_constraint_violations": self.std_constraint_violations,
            "avg_collisions": self.avg_collisions,
            "avg_joint_limit_violations": self.avg_joint_limit_violations,
            "avg_excessive_force": self.avg_excessive_force,
            "attack_success_rate": self.attack_success_rate,
            "num_baseline_success": self.num_baseline_success,
            "num_flipped": self.num_flipped,
            "avg_token_edit_ratio": self.avg_token_edit_ratio,
            "avg_char_edit_dist": self.avg_char_edit_dist,
            "avg_stealth_penalty": self.avg_stealth_penalty,
            "avg_visual_linf": self.avg_visual_linf,
            "avg_visual_pixel_change": self.avg_visual_pixel_change,
            "avg_visual_ssim": self.avg_visual_ssim,
            "avg_step_ratio": self.avg_step_ratio,
            "avg_baseline_steps": self.avg_baseline_steps,
            "avg_action_divergence": self.avg_action_divergence,
            "avg_reward": self.avg_reward,
            "std_reward": self.std_reward,
            "baseline_success_rate": self.baseline_success_rate,
        }
        for family, frac in self.tool_family_fractions.items():
            d[f"tool_family_{family}"] = frac
        return d


@dataclass
class AttackMetricsSummary:
    """Lightweight per-task or per-objective summary."""

    count: int = 0
    task_execution_rate: float = 0.0
    attack_success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_tools_called: float = 0.0
    avg_chars_changed: float = 0.0
    avg_action_seq_length: float = 0.0
    avg_constraint_violations: float = 0.0


# ============================================================================
# Helpers
# ============================================================================


def _safe_mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _safe_std(vals: Sequence[float]) -> float:
    return float(np.std(vals)) if vals else 0.0


def _ci95(vals: Sequence[float]) -> float:
    """95% confidence interval half-width (assumes normal)."""
    if len(vals) < 2:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


_TOOL_FAMILIES = {"token", "char", "prompt", "visual"}


def _classify_tool_family(tool_name: str) -> Optional[str]:
    """Map a tool name to its family (token / char / prompt / visual)."""
    t = tool_name.lower().strip()
    if not t:
        return None
    for family in _TOOL_FAMILIES:
        if family in t:
            return family
    return "other"


# ============================================================================
# Core computation
# ============================================================================


def compute_metrics_from_trajectories(
    trajectories: Sequence[Any],
) -> AttackMetrics:
    """Compute aggregate metrics from a flat list of ART trajectories.

    Each trajectory is expected to have ``.reward``, ``.metrics``, and
    ``.metadata`` populated by ``vla_attack_rollout``.
    """
    if not trajectories:
        return AttackMetrics()

    # Per-episode accumulators
    tools_called: List[float] = []
    chars_changed: List[float] = []
    token_edit_ratios: List[float] = []
    action_lengths: List[float] = []
    baseline_steps_list: List[float] = []
    step_ratios: List[float] = []
    constraint_violations: List[float] = []
    collisions: List[float] = []
    joint_limits: List[float] = []
    excessive_forces: List[float] = []
    rewards: List[float] = []
    attack_successes: List[int] = []
    baseline_successes: List[int] = []
    flips: List[int] = []
    stealth_penalties: List[float] = []
    char_edit_dists: List[float] = []
    visual_linf_list: List[float] = []
    visual_pixel_list: List[float] = []
    visual_ssim_list: List[float] = []
    action_divergences: List[float] = []

    tool_family_counter: Counter = Counter()
    per_task_data: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for traj in trajectories:
        m = getattr(traj, "metrics", {}) or {}
        md = getattr(traj, "metadata", {}) or {}

        # --- 1. Tools called ---
        n_tools = m.get("num_tool_calls", 0)
        tools_str = md.get("tools_used", "")
        if n_tools == 0 and tools_str:
            n_tools = len([t for t in tools_str.split(",") if t.strip()])
        tools_called.append(float(n_tools))

        # Tool family distribution
        if tools_str:
            for t in tools_str.split(","):
                fam = _classify_tool_family(t)
                if fam:
                    tool_family_counter[fam] += 1

        # --- 2. Characters changed ---
        orig = md.get(
            "original_instruction",
            md.get("instruction", m.get("original_instruction", "")),
        )
        pert = md.get("perturbed_instruction", "")
        if orig and pert and orig != pert:
            cd = float(edit_distance(orig, pert))
            chars_changed.append(cd)
            token_edit_ratios.append(token_edit_ratio(orig, pert))
        else:
            chars_changed.append(0.0)
            token_edit_ratios.append(0.0)

        # --- 3. Action sequence length ---
        atk_steps = float(m.get("attack_steps", 0))
        action_lengths.append(atk_steps)

        # --- 9. Step ratio ---
        base_steps = float(m.get("baseline_steps", 0))
        baseline_steps_list.append(base_steps)
        if base_steps > 0:
            step_ratios.append(atk_steps / base_steps)

        # --- 4. Task execution ---
        atk_success = int(m.get("attack_success", 0))
        base_success = int(m.get("baseline_success", 0))
        attack_successes.append(atk_success)
        baseline_successes.append(base_success)

        # --- 6. Attack flip ---
        flipped = 1 if base_success and not atk_success else 0
        flips.append(flipped)

        # --- 5. Constraint violations ---
        col = float(m.get("attack_collision_count", m.get("collision_count", 0)))
        jl = float(m.get(
            "attack_joint_limit_violations",
            m.get("joint_limit_violations", 0),
        ))
        ef = float(m.get(
            "attack_excessive_force_count",
            m.get("excessive_force_count", 0),
        ))
        total_cv = col + jl + ef
        collisions.append(col)
        joint_limits.append(jl)
        excessive_forces.append(ef)
        constraint_violations.append(total_cv)

        # --- 7. Stealth metrics (from reward computation) ---
        stealth_penalties.append(float(m.get("stealth_penalty", 0.0)))
        char_edit_dists.append(float(m.get("stealth_char_edit_dist", 0.0)))
        if "stealth_linf" in m:
            visual_linf_list.append(float(m["stealth_linf"]))
        if "stealth_pixel_change_ratio" in m:
            visual_pixel_list.append(float(m["stealth_pixel_change_ratio"]))
        if "stealth_ssim" in m:
            visual_ssim_list.append(float(m["stealth_ssim"]))

        # --- 10. Action divergence (L2 between baseline and attack action norms) ---
        base_action_jerk = float(m.get("baseline_action_jerk", 0.0))
        atk_action_jerk = float(m.get("attack_action_jerk", 0.0))
        if base_action_jerk > 0 or atk_action_jerk > 0:
            action_divergences.append(abs(atk_action_jerk - base_action_jerk))

        rewards.append(float(getattr(traj, "reward", 0.0)))

        # --- 11. Per-task grouping ---
        task_key = f"{md.get('task_suite', '?')}/task_{md.get('task_id', '?')}"
        per_task_data[task_key].append({
            "reward": float(getattr(traj, "reward", 0.0)),
            "attack_success": atk_success,
            "baseline_success": base_success,
            "flipped": flipped,
            "tools_called": float(n_tools),
            "chars_changed": chars_changed[-1],
            "action_length": atk_steps,
            "constraint_violations": total_cv,
        })

    n = len(trajectories)
    n_baseline_success = sum(baseline_successes)

    # Tool family fractions
    total_tools = sum(tool_family_counter.values()) or 1
    tool_family_fracs = {
        fam: cnt / total_tools for fam, cnt in tool_family_counter.items()
    }

    # Per-task summaries
    per_task_summaries: Dict[str, AttackMetricsSummary] = {}
    for task_key, entries in per_task_data.items():
        tc = len(entries)
        bs = sum(e["baseline_success"] for e in entries)
        fl = sum(e["flipped"] for e in entries)
        per_task_summaries[task_key] = AttackMetricsSummary(
            count=tc,
            task_execution_rate=(
                sum(e["attack_success"] for e in entries) / tc if tc else 0.0
            ),
            attack_success_rate=fl / bs if bs else 0.0,
            avg_reward=_safe_mean([e["reward"] for e in entries]),
            avg_tools_called=_safe_mean([e["tools_called"] for e in entries]),
            avg_chars_changed=_safe_mean([e["chars_changed"] for e in entries]),
            avg_action_seq_length=_safe_mean([e["action_length"] for e in entries]),
            avg_constraint_violations=_safe_mean(
                [e["constraint_violations"] for e in entries]
            ),
        )

    return AttackMetrics(
        num_episodes=n,
        # 1
        avg_tools_called=_safe_mean(tools_called),
        std_tools_called=_safe_std(tools_called),
        # 2
        avg_chars_changed=_safe_mean(chars_changed),
        std_chars_changed=_safe_std(chars_changed),
        # 3
        avg_action_seq_length=_safe_mean(action_lengths),
        std_action_seq_length=_safe_std(action_lengths),
        # 4
        task_execution_rate=sum(attack_successes) / n if n else 0.0,
        # 5
        avg_constraint_violations=_safe_mean(constraint_violations),
        std_constraint_violations=_safe_std(constraint_violations),
        avg_collisions=_safe_mean(collisions),
        avg_joint_limit_violations=_safe_mean(joint_limits),
        avg_excessive_force=_safe_mean(excessive_forces),
        # 6
        attack_success_rate=(
            sum(flips) / n_baseline_success if n_baseline_success else 0.0
        ),
        num_baseline_success=n_baseline_success,
        num_flipped=sum(flips),
        # 7
        avg_token_edit_ratio=_safe_mean(token_edit_ratios),
        avg_char_edit_dist=_safe_mean(char_edit_dists),
        avg_stealth_penalty=_safe_mean(stealth_penalties),
        avg_visual_linf=_safe_mean(visual_linf_list),
        avg_visual_pixel_change=_safe_mean(visual_pixel_list),
        avg_visual_ssim=_safe_mean(visual_ssim_list),
        # 8
        tool_family_counts=dict(tool_family_counter),
        tool_family_fractions=tool_family_fracs,
        # 9
        avg_step_ratio=_safe_mean(step_ratios),
        avg_baseline_steps=_safe_mean(baseline_steps_list),
        # 10
        avg_action_divergence=_safe_mean(action_divergences),
        # general
        avg_reward=_safe_mean(rewards),
        std_reward=_safe_std(rewards),
        baseline_success_rate=n_baseline_success / n if n else 0.0,
        # 11
        per_task=per_task_summaries,
        # raw
        per_episode={
            "tools_called": tools_called,
            "chars_changed": chars_changed,
            "token_edit_ratios": token_edit_ratios,
            "action_seq_length": action_lengths,
            "step_ratios": step_ratios,
            "constraint_violations": constraint_violations,
            "rewards": rewards,
            "flipped": [float(f) for f in flips],
        },
    )


def compute_metrics(groups: Sequence[Any]) -> AttackMetrics:
    """Compute aggregate metrics from ART trajectory groups.

    Parameters
    ----------
    groups : sequence of TrajectoryGroup
        As returned by ``art.gather_trajectory_groups``.
    """
    all_trajs = []
    for g in groups:
        trajs = getattr(g, "trajectories", [])
        all_trajs.extend(trajs)
    return compute_metrics_from_trajectories(all_trajs)


# ============================================================================
# Formatting / printing
# ============================================================================


def print_metrics(
    m: AttackMetrics,
    step: Optional[int] = None,
    logger_fn=None,
) -> str:
    """Format metrics as a human-readable table string.

    Also logs via ``logger_fn`` if provided (e.g. ``logger.info``).
    Returns the formatted string.
    """
    header = "Attack Evaluation Metrics"
    if step is not None:
        header += f" (step {step})"

    sep = "─" * 52
    lines = [
        f"{'=' * 56}",
        f"  {header}",
        f"  {sep}",
        f"  Episodes evaluated:          {m.num_episodes}",
        f"  {sep}",
        # Paper metrics
        f"  Avg tools called:            {m.avg_tools_called:.2f} ± {m.std_tools_called:.2f}",
        f"  Avg chars changed:           {m.avg_chars_changed:.1f} ± {m.std_chars_changed:.1f}",
        f"  Avg action seq length:       {m.avg_action_seq_length:.1f} ± {m.std_action_seq_length:.1f}",
        f"  Task execution rate:         {m.task_execution_rate:.1%}",
        f"  Avg constraint violations:   {m.avg_constraint_violations:.2f} ± {m.std_constraint_violations:.2f}",
        f"    ├ collisions:              {m.avg_collisions:.2f}",
        f"    ├ joint-limit hits:        {m.avg_joint_limit_violations:.2f}",
        f"    └ excessive-force events:  {m.avg_excessive_force:.2f}",
        f"  {sep}",
        # Attack effectiveness
        f"  Attack success rate (ASR):   {m.attack_success_rate:.1%}  ({m.num_flipped}/{m.num_baseline_success} flipped)",
        f"  Baseline success rate:       {m.baseline_success_rate:.1%}",
        f"  Avg reward:                  {m.avg_reward:.3f} ± {m.std_reward:.3f}",
        f"  {sep}",
        # Perturbation
        f"  Avg token edit ratio:        {m.avg_token_edit_ratio:.3f}",
        f"  Avg char edit distance:      {m.avg_char_edit_dist:.1f}",
        f"  Avg stealth penalty:         {m.avg_stealth_penalty:.3f}",
    ]

    if m.avg_visual_linf > 0:
        lines.extend([
            f"  Avg visual L∞:               {m.avg_visual_linf:.1f}",
            f"  Avg pixel change ratio:      {m.avg_visual_pixel_change:.4f}",
            f"  Avg SSIM:                    {m.avg_visual_ssim:.4f}",
        ])

    lines.append(f"  {sep}")

    # Step ratio
    lines.append(
        f"  Avg step ratio (atk/base):   {m.avg_step_ratio:.2f}  "
        f"(base avg: {m.avg_baseline_steps:.0f})"
    )

    if m.avg_action_divergence > 0:
        lines.append(
            f"  Avg action divergence:       {m.avg_action_divergence:.4f}"
        )

    # Tool family breakdown
    if m.tool_family_fractions:
        lines.append(f"  {sep}")
        lines.append("  Tool family distribution:")
        for fam in sorted(m.tool_family_fractions.keys()):
            cnt = m.tool_family_counts.get(fam, 0)
            pct = m.tool_family_fractions[fam]
            lines.append(f"    {fam:<12}  {cnt:>4} calls  ({pct:.0%})")

    # Per-task breakdown (compact)
    if m.per_task:
        lines.append(f"  {sep}")
        lines.append("  Per-task breakdown:")
        lines.append(
            f"    {'Task':<28} {'N':>3} {'ASR':>6} "
            f"{'TER':>6} {'Reward':>7} {'Tools':>5}"
        )
        for task_key in sorted(m.per_task.keys()):
            s = m.per_task[task_key]
            lines.append(
                f"    {task_key:<28} {s.count:>3} "
                f"{s.attack_success_rate:>5.0%} "
                f"{s.task_execution_rate:>5.0%} "
                f"{s.avg_reward:>+7.3f} "
                f"{s.avg_tools_called:>5.1f}"
            )

    lines.append(f"{'=' * 56}")
    text = "\n".join(lines)

    if logger_fn:
        logger_fn(text)
    return text


def metrics_to_latex_row(
    m: AttackMetrics,
    label: str = "",
) -> str:
    """Format key metrics as a LaTeX table row for paper inclusion.

    Columns: Label & Tools & Chars & ActLen & TER & CV & ASR
    """
    return (
        f"{label} & "
        f"{m.avg_tools_called:.1f} & "
        f"{m.avg_chars_changed:.1f} & "
        f"{m.avg_action_seq_length:.0f} & "
        f"{m.task_execution_rate:.1%} & "
        f"{m.avg_constraint_violations:.1f} & "
        f"{m.attack_success_rate:.1%} \\\\"
    )
