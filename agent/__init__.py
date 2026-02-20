"""Agent package exports.

This package supports two independent rollouts:
 - HotpotQA (text-only) in `agent.rollout` (optional; requires HotpotQA dataset code)
 - LIBERO / π0.5 VLA in `agent.vla_rollout`

We intentionally do NOT import `agent.rollout` here, so running LIBERO/VLA does
not require any HotpotQA-specific dependencies to be installed.
"""

from .vla_rollout import (
    ToolSet,
    VLAAttackScenario,
    VLAAttackState,
    build_vla_attack_tools,
    get_vla_model,
    set_vla_model,
    vla_attack_rollout,
)

__all__ = [
    "ToolSet",
    "VLAAttackScenario",
    "VLAAttackState",
    "build_vla_attack_tools",
    "get_vla_model",
    "set_vla_model",
    "vla_attack_rollout",
]