"""
Model registry for LIBERO evaluation.

Supports OpenPI pi0 and pi0.5 natively. Other models (DeepThinkVLA, MolmoAct,
ECoT, InternVLA-M1, OpenVLA, StarVLA, X-VLA, LightVLA) are documented in
eval/README.md with their repo eval commands; this registry only loads
openpi_pi0 and openpi_pi05 for run_libero_eval.py.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Default hyperparameters per model (from official sources / OpenPI config).
# OpenPI: Physical-Intelligence/openpi config.py - pi0_libero, pi05_libero use action_horizon=10.
# replan_steps: steps executed from each action chunk before re-planning (we use 5 for OpenPI).
# External models: episodes_per_task/seed match eval/external/configs; action_* only used if we add native wrappers.
MODEL_DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "openpi_pi0": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "openpi_pi05": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "openvla": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "xvla": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "molmoact": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "deepthinkvla": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "ecot": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "internvla_m1": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "starvla": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
    "lightvla": {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42},
}


def get_model_defaults(model_id: str) -> Dict[str, Any]:
    """Return default hyperparameters for a model (action_horizon, replan_steps, episodes_per_task, seed)."""
    key = model_id.lower().replace("-", "_")
    return dict(MODEL_DEFAULT_HYPERPARAMS.get(key, {"action_horizon": 10, "replan_steps": 5, "episodes_per_task": 5, "seed": 42}))

# Suite name alias: "long" -> libero_10
SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}

# Max steps per suite (same as vla_rollout)
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Models that run_libero_eval.py can load directly
NATIVE_MODELS = ("openpi_pi0", "openpi_pi05")


def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    action_horizon: Optional[int] = None,
    replan_steps: Optional[int] = None,
) -> Any:
    """Load a VLA model for LIBERO. Returns an object with set_language(), predict()."""
    defaults = get_model_defaults(model_name)
    if action_horizon is None:
        action_horizon = defaults["action_horizon"]
    if replan_steps is None:
        replan_steps = defaults["replan_steps"]
    if model_name == "openpi_pi0":
        from libero_rollouts.pi0_libero_model import Pi0LiberoModel
        return Pi0LiberoModel(
            train_config_name="pi0_libero",
            checkpoint_path=checkpoint_path,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )
    if model_name == "openpi_pi05":
        from libero_rollouts.pi05_libero_model import Pi05LiberoModel
        return Pi05LiberoModel(
            train_config_name="pi05_libero",
            checkpoint_path=checkpoint_path,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )
    raise ValueError(
        f"Unknown model '{model_name}'. Supported: {list(NATIVE_MODELS)}. "
        "For other models (DeepThinkVLA, MolmoAct, etc.) see eval/README.md."
    )


def make_policy_fn(
    model: Any,
    instruction: str,
    replan_steps: int,
) -> Callable[[dict, str], Tuple[np.ndarray, str]]:
    """Build policy_fn(obs, instruction) -> (action, reasoning) for collect_libero_rollout_info."""
    from libero_rollouts.pi05_libero_model import preprocess_image, build_libero_state

    action_buffer: list = []
    current_instruction = instruction

    def policy_fn(obs: dict, instr: str) -> Tuple[np.ndarray, str]:
        nonlocal action_buffer, current_instruction
        if instr != current_instruction:
            current_instruction = instr
        if not action_buffer:
            agentview = preprocess_image(obs["agentview_image"])
            wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
            state = build_libero_state(obs)
            model.set_language(current_instruction)
            actions = model.predict(agentview, wrist, state)
            action_buffer.extend(actions[:replan_steps].tolist())
        action = np.array(action_buffer.pop(0), dtype=np.float64)
        return action, ""

    return policy_fn
