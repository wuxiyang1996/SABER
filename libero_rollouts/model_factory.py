"""Unified VLA model loading for LIBERO evaluation.

Maps model_id -> VLA wrapper class. All wrappers implement:
    set_language(instruction: str) -> None
    predict(agentview_224: np.ndarray, wrist_224: np.ndarray, state_8: np.ndarray) -> np.ndarray
    predict_from_obs(obs: dict) -> np.ndarray

The factory handles per-suite checkpoints (OpenVLA, LightVLA, MolmoAct, etc.)
and returns a model ready for LIBERO rollouts.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Per-suite HuggingFace checkpoint mapping
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
    "lightvla": {
        "libero_spatial": "TTJiang/LightVLA-libero-spatial",
        "libero_object": "TTJiang/LightVLA-libero-object",
        "libero_goal": "TTJiang/LightVLA-libero-goal",
        "libero_10": "TTJiang/LightVLA-libero-10",
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
    "xvla": {
        "_all": "2toINF/X-VLA-Libero",
    },
    "openvla_oft": {
        "_all": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    },
    "smolvla": {
        "_all": "HuggingFaceVLA/smolvla_libero",
    },
    "groot": {
        "libero_spatial": "Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL-8K",
        "libero_object": "Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT-8K",
        "libero_goal": "Tacoin/GR00T-N1.5-3B-LIBERO-GOAL-8K",
        "libero_10": "Tacoin/GR00T-N1.5-3B-LIBERO-LONG-8K",
    },
    "inspirevla": {
        "_all": "InspireVLA/minivla-inspire-libero-union4",
    },
    "minivla": {
        "_all": "Stanford-ILIAD/minivla-libero90-prismatic",
    },
    "go1": {
        "_all": "127.0.0.1:9000",  # AgiBot-World GO-1 server URL (override with GO1_SERVER env)
    },
}


# Model action horizons: how many actions returned per inference call
_MODEL_ACTION_HORIZONS = {
    "openpi_pi0": 10,
    "openpi_pi05": 10,
    "openvla": 1,
    "ecot": 1,
    "lightvla": 1,
    "deepthinkvla": 10,
    "molmoact": 1,
    "internvla_m1": 8,
    "xvla": 30,
    "openvla_oft": 8,
    "smolvla": 1,
    "groot": 16,
    "inspirevla": 1,
    "minivla": 1,
    "go1": 5,
}

# Aliases for victim/model names (e.g. open_pi0.5 -> openpi_pi05)
_MODEL_ID_ALIASES = {
    "open_pi0": "openpi_pi0",
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
    "openvla", "lightvla", "ecot", "deepthinkvla", "deepthinkvla_eval",
    "molmoact", "internvla_m1", "xvla", "openvla_oft", "smolvla", "groot",
    "inspirevla", "minivla",
}

_ABSOLUTE_ACTION_MODELS = {"xvla"}

# Models that must use predict_from_obs() instead of shared preprocess_image+predict().
# Each has model-specific preprocessing (image size, JPEG round-trip, etc.) that the
# shared 224x224 preprocess_image() pipeline would break.
_OBS_PREDICT_MODELS = {
    "xvla",       # absolute EE actions, needs raw obs for controller state
    "smolvla",    # trained at 256x256 (not 224x224)
    "openvla_oft", # needs JPEG round-trip to match RLDS training pipeline
    "groot",      # trained at 256x256, needs raw obs for EEF decomposition
    "go1",        # AgiBot client: send 256x256 images to match their eval
    "inspirevla", # openvla-mini uses flipud, not 180° rotation
    "minivla",    # openvla-mini uses flipud, not 180° rotation
}


_MODEL_ENV_MAP = {
    "openvla": "vla_models",
    "lightvla": "vla_models",
    "ecot": "vla_models",
    "deepthinkvla": "vla_deepthinkvla",
    "deepthinkvla_eval": "vla_deepthinkvla",
    "molmoact": "vla_molmoact",
    "internvla_m1": "vla_internvla",
    "xvla": "vla_xvla",
    "openvla_oft": "vla_models",
    "smolvla": "vla_smolvla",
    "groot": "vla_smolvla",
    "inspirevla": "vla_inspirevla",
    "minivla": "vla_inspirevla",
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
        Model identifier (e.g. "openpi_pi05", "openvla", "lightvla").
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

    if key == "openpi_pi0":
        from pi0_libero_model import Pi0LiberoModel
        return Pi0LiberoModel(
            train_config_name="pi0_libero",
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
        wrapper = SubprocessVLAWrapper(
            python=vla_python,
            model_id=key,
            suite_name=suite_name,
            checkpoint=ckpt,
            device=device,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
        )
        if key in _ABSOLUTE_ACTION_MODELS:
            wrapper.uses_absolute_actions = True
        if key in _OBS_PREDICT_MODELS:
            wrapper.use_obs_predict = True
        return wrapper

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

    if key == "lightvla":
        from lightvla_wrapper import LightVLAWrapper
        return LightVLAWrapper(
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

    if key == "xvla":
        from xvla_wrapper import XVLAWrapper
        return XVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "openvla_oft":
        from openvla_oft_wrapper import OpenVLAOFTWrapper
        return OpenVLAOFTWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "smolvla":
        from smolvla_wrapper import SmolVLAWrapper
        return SmolVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "groot":
        from groot_wrapper import GR00TWrapper
        return GR00TWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key in ("inspirevla", "minivla"):
        from inspirevla_wrapper import InspireVLAWrapper
        return InspireVLAWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    if key == "go1":
        from go1_client_wrapper import GO1ClientWrapper
        return GO1ClientWrapper(
            checkpoint=ckpt, suite_name=suite_name, device=device,
            action_horizon=action_horizon, replan_steps=replan_steps,
        )

    raise ValueError(
        f"Unknown model: {model_id}. Supported: openpi_pi0, openpi_pi05, "
        "openvla, openvla_oft, ecot, lightvla, deepthinkvla, molmoact, internvla_m1, "
        "xvla, smolvla, groot, inspirevla, minivla, go1"
    )


def is_per_suite_model(model_id: str) -> bool:
    """True if the model uses per-suite checkpoints (must reload per suite)."""
    key = _normalize_model_id(model_id)
    if key in ("openpi_pi0", "openpi_pi05"):
        return False
    ckpts = _SUITE_CHECKPOINTS.get(key, {})
    return "_all" not in ckpts and len(ckpts) > 1


def is_jax_model(model_id: str) -> bool:
    return _normalize_model_id(model_id) in ("openpi_pi0", "openpi_pi05")


def is_absolute_action_model(model_id: str) -> bool:
    """True if the model outputs absolute EE targets (not deltas)."""
    return _normalize_model_id(model_id) in _ABSOLUTE_ACTION_MODELS
