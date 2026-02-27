"""
Pi0 Model Wrapper for LIBERO Benchmark.

Wraps the OpenPI pi0 (flow-matching VLA) model for inference in the LIBERO
manipulation benchmark. Uses the same observation/action interface as
Pi05LiberoModel (agentview + wrist images, 8-dim state, 7-dim actions).

By default loads the pi0_base checkpoint for zero-shot LIBERO evaluation.
To use a LIBERO-finetuned pi0 checkpoint, pass checkpoint_path (e.g. from
training with the OpenPI pi0_libero config).

LIBERO observations:
  - agentview_image, robot0_eye_in_hand_image, robot0_eef_pos,
    robot0_eef_quat, robot0_gripper_qpos

LIBERO actions: 7-dim (xyz_delta + rpy_delta + gripper)
"""

import os
import sys

# Reuse path setup and helpers from pi05 (same openpi package)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_THIS_DIR, ".."))

if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

# Ensure libero_rollouts and openpi are on path (same as pi05)
_LIBERO_ROLLOUTS = os.path.join(_FRAMEWORK_ROOT, "libero_rollouts")
_INREPO_OPENPI_SRC = os.path.join(_FRAMEWORK_ROOT, "openpi", "src")
_ROBOTWIN_ROOT = os.environ.get(
    "ROBOTWIN_ROOT",
    os.path.realpath(os.path.join(_THIS_DIR, "..", "..", "RoboTwin")),
)
_PI05_POLICY_DIR = os.path.join(_ROBOTWIN_ROOT, "policy", "pi05")
_PI05_SRC_DIR = os.path.join(_PI05_POLICY_DIR, "src")

_use_inrepo = (
    os.environ.get("ROBOTWIN_ROOT") is None
    and os.path.isdir(_INREPO_OPENPI_SRC)
    and os.path.isdir(os.path.join(_INREPO_OPENPI_SRC, "openpi"))
)
if _use_inrepo:
    if _INREPO_OPENPI_SRC not in sys.path:
        sys.path.insert(0, _INREPO_OPENPI_SRC)
    _INREPO_OPENPI_CLIENT_SRC = os.path.join(
        _FRAMEWORK_ROOT, "openpi", "packages", "openpi-client", "src"
    )
    if os.path.isdir(_INREPO_OPENPI_CLIENT_SRC) and _INREPO_OPENPI_CLIENT_SRC not in sys.path:
        sys.path.insert(0, _INREPO_OPENPI_CLIENT_SRC)
else:
    if os.path.isdir(_ROBOTWIN_ROOT):
        for p in (_PI05_POLICY_DIR, _PI05_SRC_DIR):
            if p not in sys.path:
                sys.path.insert(0, p)

if _LIBERO_ROLLOUTS not in sys.path:
    sys.path.insert(0, _LIBERO_ROLLOUTS)

# Reuse helpers from pi05 (same preprocessing and state format)
from pi05_libero_model import (  # noqa: E402
    build_libero_state,
    preprocess_image,
)

import numpy as np


# ---------------------------------------------------------------------------
# Pi0 LIBERO model
# ---------------------------------------------------------------------------
class Pi0LiberoModel:
    """High-level wrapper around the OpenPI pi0 policy for LIBERO evaluation.

    Same API as Pi05LiberoModel: agentview + wrist images, 8-dim state,
    7-dim actions. Uses OpenPI config ``pi0_libero`` (LIBERO observation/action
    space). Default checkpoint is pi0_base (zero-shot); pass checkpoint_path
    for a LIBERO-finetuned pi0 if available.
    """

    # Zero-shot: base pi0 checkpoint. For finetuned LIBERO pi0, pass checkpoint_path.
    DEFAULT_CHECKPOINT_URL = "gs://openpi-assets/checkpoints/pi0_base"

    def __init__(
        self,
        train_config_name: str = "pi0_libero",
        checkpoint_path: str | None = None,
        asset_id: str | None = None,
        action_horizon: int = 10,
        replan_steps: int = 5,
    ):
        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.shared import download as _download
            from openpi.training import config as _config
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Cannot import openpi: {e}. Pi0 wrapper requires openpi "
                "(in-repo openpi/src or RoboTwin policy/pi05). See INSTALL.md."
            ) from e

        self.train_config_name = train_config_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps

        ckpt_url = checkpoint_path if checkpoint_path else self.DEFAULT_CHECKPOINT_URL
        print(f"[Pi0LiberoModel] Loading checkpoint from {ckpt_url} ...")
        ckpt_dir = str(_download.maybe_download(ckpt_url))

        assets_dir = os.path.join(ckpt_dir, "assets")
        resolved_asset_id = asset_id
        if resolved_asset_id is None and os.path.isdir(assets_dir):
            for root, _dirs, files in os.walk(assets_dir):
                if "norm_stats.json" in files:
                    resolved_asset_id = os.path.relpath(root, assets_dir)
                    break

        config = _config.get_config(self.train_config_name)
        norm_stats = None
        if resolved_asset_id:
            try:
                from openpi.training import checkpoints as _checkpoints
                norm_stats = _checkpoints.load_norm_stats(
                    os.path.join(ckpt_dir, "assets"), resolved_asset_id
                )
                print(f"[Pi0LiberoModel] Loaded norm stats for asset '{resolved_asset_id}'")
            except Exception as e:
                print(f"[Pi0LiberoModel] Norm stats load failed: {e}, using config defaults.")
                norm_stats = None

        self.policy = _policy_config.create_trained_policy(
            config, ckpt_dir, norm_stats=norm_stats
        )
        print(f"[Pi0LiberoModel] Loaded checkpoint from {ckpt_dir}")
        self.instruction: str | None = None

    def set_language(self, instruction: str) -> None:
        if instruction != self.instruction:
            self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        assert self.instruction is not None, "Call set_language() before predict()."
        obs_dict = {
            "observation/image": agentview_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state.astype(np.float32),
            "prompt": self.instruction,
        }
        raw_actions = self.policy.infer(obs_dict)["actions"]
        actions = np.asarray(raw_actions)[: self.action_horizon]
        return actions

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image(obs["agentview_image"])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
