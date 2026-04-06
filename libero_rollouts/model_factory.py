"""Unified VLA model loading for LIBERO evaluation.

Maps model_id -> VLA wrapper class. All wrappers implement:
    set_language(instruction: str) -> None
    predict(agentview_224: np.ndarray, wrist_224: np.ndarray, state_8: np.ndarray) -> np.ndarray
    predict_from_obs(obs: dict) -> np.ndarray

Supported models (paper): Pi0.5, OpenVLA, ECoT, DeepThinkVLA, MolmoAct, InternVLA-M1.
The factory handles per-suite checkpoints and returns a model ready for LIBERO rollouts.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_SUITE_CHECKPOINTS = {
    "openvla": {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object": "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    },
    "ecot": {
        "_all": "Embodied-CoT/ecot-openvla-7b-bridge",
    },
    "deepthinkvla": {
        "_all": "yinchenghust/deepthinkvla_libero_cot_rl",
    },
    "molmoact": {
        "libero_spatial": "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812",
        "libero_object": "allenai/MolmoAct-7B-D-LIBERO-Object-0812",
        "libero_goal": "allenai/MolmoAct-7B-D-LIBERO-Goal-0812",
        "libero_10": "allenai/MolmoAct-7B-D-LIBERO-Long-0812",
    },
    "internvla_m1": {
        "libero_spatial": "InternRobotics/InternVLA-M1-LIBERO-Spatial",
        "libero_object": "InternRobotics/InternVLA-M1-LIBERO-Object",
        "libero_goal": "InternRobotics/InternVLA-M1-LIBERO-Goal",
        "libero_10": "InternRobotics/InternVLA-M1-LIBERO-Long",
    },
}


# Model action horizons: how many actions returned per inference call
_MODEL_ACTION_HORIZONS = {
    "openpi_pi05": 10,
    "openvla": 1,
    "ecot": 1,
    "deepthinkvla": 10,
    "molmoact": 1,
    "internvla_m1": 8,
}

# Aliases for victim/model names (e.g. open_pi0.5 -> openpi_pi05)
_MODEL_ID_ALIASES = {
    "open_pi0.5": "openpi_pi05",
    "openpi_pi0.5": "openpi_pi05",
}


def _normalize_model_id(model_id: str) -> str:
    """Normalize model_id to canonical key used in this module."""
    key = model_id.lower().replace("-", "_")
    return _MODEL_ID_ALIASES.get(key, key)


def get_checkpoint_for_suite(model_id: str, suite_name: str) -> str:
    """Return the HuggingFace checkpoint ID for a given model and LIBERO suite."""
    ckpts = _SUITE_CHECKPOINTS.get(_normalize_model_id(model_id), {})
    if "_all" in ckpts:
        return ckpts["_all"]
    if suite_name in ckpts:
        return ckpts[suite_name]
    raise ValueError(
        f"No checkpoint mapping for model={model_id}, suite={suite_name}. "
        f"Available: {list(ckpts.keys())}"
    )


def get_action_horizon(model_id: str) -> int:
    return _MODEL_ACTION_HORIZONS.get(_normalize_model_id(model_id), 1)


_SUBPROCESS_MODELS = {
    "openvla", "ecot", "deepthinkvla", "deepthinkvla_eval",
    "molmoact", "internvla_m1",
}


_MODEL_ENV_MAP = {
    "openvla": "vla_models",
    "ecot": "vla_models",
    "deepthinkvla": "vla_deepthinkvla",
    "deepthinkvla_eval": "vla_deepthinkvla",
    "molmoact": "vla_molmoact",
    "internvla_m1": "vla_internvla",
}


def _get_vla_python(model_id: Optional[str] = None) -> Optional[str]:
    """Return path to the VLA-env Python interpreter, or None.

    Resolution order:
    1. VLA_PYTHON env var (explicit override)
    2. Per-model env from _MODEL_ENV_MAP (model-specific conda env)
    3. Default vla_models env
    """
    p = os.environ.get("VLA_PYTHON")
    if p:
        return p

    conda_base = os.environ.get(
        "CONDA_PREFIX_BASE",
        "/workspace/miniforge3",
    )
    # strip envs/xxx if CONDA_PREFIX points into an env
    if "/envs/" in conda_base:
        conda_base = conda_base.split("/envs/")[0]

    if model_id:
        key = _normalize_model_id(model_id)
        env_name = _MODEL_ENV_MAP.get(key)
        if env_name:
            candidate = os.path.join(conda_base, "envs", env_name, "bin", "python")
            if os.path.isfile(candidate):
                return candidate

    default = os.path.join(conda_base, "envs", "vla_models", "bin", "python")
    if os.path.isfile(default):
        return default
    return None


def load_vla_model(
    model_id: str,
    suite_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:0",
    action_horizon: Optional[int] = None,
    replan_steps: int = 5,
) -> Any:
    """Load a VLA model for LIBERO evaluation.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. "openpi_pi05", "openvla", "ecot").
    suite_name : str, optional
        LIBERO suite name for per-suite checkpoint selection.
    checkpoint_path : str, optional
        Override checkpoint path/HF repo ID.
    device : str
        PyTorch device string (ignored for JAX models).
    action_horizon : int, optional
        Number of actions per inference call (auto-detected if None).
    replan_steps : int
        Steps from each action chunk to use before re-planning.
    """
    key = _normalize_model_id(model_id)
    if action_horizon is None:
        action_horizon = get_action_horizon(key)

    # --- Native OpenPI models (JAX) — always run in-process ---
    if key == "openpi_pi05":
        from pi05_libero_model import Pi05LiberoModel
        return Pi05LiberoModel(
            train_config_name="pi05_libero",
            checkpoint_path=checkpoint_path,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )

    # --- Subprocess mode: run HF models in an isolated env ---
    # Skip if we are already inside a subprocess server (prevents infinite recursion).
    _in_subprocess = os.environ.get("_VLA_SUBPROCESS_SERVER") == "1"
    vla_python = _get_vla_python(model_id=key) if not _in_subprocess else None
    if vla_python and key in _SUBPROCESS_MODELS:
        from subprocess_vla_wrapper import SubprocessVLAWrapper
        ckpt = checkpoint_path or get_checkpoint_for_suite(key, suite_name)
        return SubprocessVLAWrapper(
            python=vla_python,
            model_id=key,
            suite_name=suite_name,
            checkpoint=ckpt,
            device=device,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )

    # --- Fallback: load in-process (same env) ---
    ckpt = checkpoint_path or get_checkpoint_for_suite(key, suite_name)

    if key == "openvla":
        from openvla_wrapper import OpenVLAWrapper
        return OpenVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "ecot":
        from ecot_wrapper import ECoTWrapper
        return ECoTWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key in ("deepthinkvla", "deepthinkvla_eval"):
        from deepthinkvla_wrapper import DeepThinkVLAWrapper
        return DeepThinkVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "molmoact":
        from molmoact_wrapper import MolmoActWrapper
        return MolmoActWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "internvla_m1":
        from internvla_wrapper import InternVLAWrapper
        return InternVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    raise ValueError(
        f"Unknown model: {model_id}. Supported: openpi_pi05, "
        "openvla, ecot, deepthinkvla, molmoact, internvla_m1"
    )


def is_per_suite_model(model_id: str) -> bool:
    """True if the model uses per-suite checkpoints (must reload per suite)."""
    key = _normalize_model_id(model_id)
    if key == "openpi_pi05":
        return False
    ckpts = _SUITE_CHECKPOINTS.get(key, {})
    return "_all" not in ckpts and len(ckpts) > 1


def is_jax_model(model_id: str) -> bool:
    return _normalize_model_id(model_id) == "openpi_pi05"
