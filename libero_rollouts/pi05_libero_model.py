"""
Pi0.5 Model Wrapper for LIBERO Benchmark.

Wraps the OpenPI pi0.5 (flow-matching VLA) model for inference
in the LIBERO manipulation benchmark.

LIBERO observations:
  - agentview_image: (H, W, 3) uint8 — third-person camera
  - robot0_eye_in_hand_image: (H, W, 3) uint8 — wrist camera
  - robot0_eef_pos: (3,) — end-effector position
  - robot0_eef_quat: (4,) — end-effector quaternion
  - robot0_gripper_qpos: (2,) — gripper joint positions

LIBERO actions: 7-dim (xyz_delta + rpy_delta + gripper)
"""

import os
import sys
import math

# Limit JAX GPU memory BEFORE any JAX/openpi imports.
# Default lowered to 0.25 so that vLLM (attack agent) can coexist on the
# same GPU.  train_vla.py sets this env-var before imports; the value here
# is only a fallback for standalone usage.
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths so the openpi library is importable. Prefer in-repo openpi
# (agent_attack_framework/openpi/src), then RoboTwin/policy/pi05.
# Set ROBOTWIN_ROOT to force the RoboTwin path when both exist.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_THIS_DIR, ".."))
_INREPO_OPENPI_SRC = os.path.join(_FRAMEWORK_ROOT, "openpi", "src")

_ROBOTWIN_ROOT = os.environ.get(
    "ROBOTWIN_ROOT",
    os.path.realpath(os.path.join(_THIS_DIR, "..", "..", "RoboTwin")),
)
_PI05_POLICY_DIR = os.path.join(_ROBOTWIN_ROOT, "policy", "pi05")
_PI05_SRC_DIR = os.path.join(_PI05_POLICY_DIR, "src")

# Prefer in-repo openpi (agent_attack_framework/openpi/src) unless ROBOTWIN_ROOT is set
_use_inrepo = (
    os.environ.get("ROBOTWIN_ROOT") is None
    and os.path.isdir(_INREPO_OPENPI_SRC)
    and os.path.isdir(os.path.join(_INREPO_OPENPI_SRC, "openpi"))
)

if _use_inrepo:
    if _INREPO_OPENPI_SRC not in sys.path:
        sys.path.insert(0, _INREPO_OPENPI_SRC)
    # openpi imports openpi_client; use in-repo package when present
    _INREPO_OPENPI_CLIENT_SRC = os.path.join(_FRAMEWORK_ROOT, "openpi", "packages", "openpi-client", "src")
    if os.path.isdir(_INREPO_OPENPI_CLIENT_SRC) and _INREPO_OPENPI_CLIENT_SRC not in sys.path:
        sys.path.insert(0, _INREPO_OPENPI_CLIENT_SRC)
else:
    if not os.path.isdir(_ROBOTWIN_ROOT):
        raise FileNotFoundError(
            f"RoboTwin root not found: {_ROBOTWIN_ROOT}\n"
            "The Pi0.5 VLA wrapper requires openpi: either use the in-repo copy at\n"
            "  agent_attack_framework/openpi/src  (no ROBOTWIN_ROOT set),\n"
            "or clone RoboTwin and set ROBOTWIN_ROOT. See INSTALL.md."
        )
    if not os.path.isdir(_PI05_POLICY_DIR):
        raise FileNotFoundError(
            f"Pi0.5 policy dir not found: {_PI05_POLICY_DIR}\n"
            "RoboTwin must contain policy/pi05 (openpi library). See INSTALL.md."
        )
    for p in (_PI05_POLICY_DIR, _PI05_SRC_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle (3,).

    Copied from robosuite for consistency with LIBERO's conventions.
    """
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def build_libero_state(obs: dict) -> np.ndarray:
    """Build the 8-dim proprioceptive state vector expected by the pi0.5 LIBERO policy.

    Returns: (8,) float32 — [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    """
    eef_pos = obs["robot0_eef_pos"]  # (3,)
    axis_angle = quat2axisangle(obs["robot0_eef_quat"])  # (3,)
    gripper = obs["robot0_gripper_qpos"]  # (2,)
    return np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float32)


def preprocess_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Rotate 180 degrees and resize to (size, size, 3) uint8.

    LIBERO images need to be rotated 180 degrees to match training preprocessing.
    """
    import cv2
    # Rotate 180 degrees (same as [::-1, ::-1]).
    img = np.ascontiguousarray(img[::-1, ::-1])
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class Pi05LiberoModel:
    """High-level wrapper around the pi0.5 policy for LIBERO evaluation.

    The model accepts:
      - agentview image (third-person camera)
      - wrist image (robot0_eye_in_hand)
      - 8-dim proprioceptive state (eef_pos + axis_angle + gripper_qpos)
      - language instruction
    and returns a sequence of 7-dim actions (xyz + rpy + gripper deltas).

    Parameters
    ----------
    train_config_name : str
        OpenPI training-config name. ``"pi05_libero"`` for the LIBERO pi0.5 config.
    checkpoint_path : str or None
        Explicit checkpoint path (local or gs://). If ``None``, auto-downloads ``pi05_base``.
    asset_id : str
        Normalization-stats asset id. ``"libero"`` for LIBERO-specific stats,
        or ``"trossen"`` if using the base model.
    action_horizon : int
        Number of action steps to execute per inference call (action chunking).
    replan_steps : int
        Number of steps from the action chunk to use before re-planning.
    """

    BASE_CHECKPOINT_URL = "gs://openpi-assets/checkpoints/pi05_libero"

    def __init__(
        self,
        train_config_name: str = "pi05_libero",
        checkpoint_path: str | None = None,
        asset_id: str | None = None,
        action_horizon: int = 50,
        replan_steps: int = 5,
    ):
        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.shared import download as _download
            from openpi.training import config as _config
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Cannot import openpi: {e}\n"
                "The openpi library is provided by the RoboTwin repo (policy/pi05).\n"
                "Ensure ROBOTWIN_ROOT points to the RoboTwin root and that policy/pi05/src "
                "contains the openpi package. See INSTALL.md in agent_attack_framework."
            ) from e

        self.train_config_name = train_config_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps

        # --- Resolve checkpoint path ----------------------------------------
        if checkpoint_path is not None:
            ckpt_dir = str(_download.maybe_download(checkpoint_path))
        else:
            print(f"[Pi05LiberoModel] Downloading Pi0.5-LIBERO from {self.BASE_CHECKPOINT_URL} ...")
            ckpt_dir = str(_download.maybe_download(self.BASE_CHECKPOINT_URL))

        # --- Determine asset_id (normalization stats) -----------------------
        assets_dir = os.path.join(ckpt_dir, "assets")
        resolved_asset_id = None
        if asset_id is not None:
            resolved_asset_id = asset_id
        elif os.path.isdir(assets_dir):
            available = [d for d in os.listdir(assets_dir)
                         if os.path.isdir(os.path.join(assets_dir, d))]
            # Prefer 'libero' if available, else 'trossen' (compatible with base model).
            if "libero" in available:
                resolved_asset_id = "libero"
            elif "trossen" in available:
                resolved_asset_id = "trossen"
            elif available:
                resolved_asset_id = available[0]
            else:
                print(f"[Pi05LiberoModel] WARNING: No assets in {assets_dir}, "
                      "norm stats will be loaded from the config.")

        # --- Load policy ----------------------------------------------------
        config = _config.get_config(self.train_config_name)

        # Try loading norm stats explicitly.  If the checkpoint doesn't have
        # LIBERO-specific stats (e.g. using the base model), let
        # create_trained_policy fall back to the config's default mechanism.
        norm_stats = None
        if resolved_asset_id is not None:
            try:
                from openpi.training import checkpoints as _checkpoints
                norm_stats = _checkpoints.load_norm_stats(
                    os.path.join(ckpt_dir, "assets"), resolved_asset_id,
                )
                print(f"[Pi05LiberoModel] Loaded norm stats for asset '{resolved_asset_id}'")
            except Exception as e:
                print(f"[Pi05LiberoModel] Could not load norm stats for "
                      f"'{resolved_asset_id}': {e}  — falling back to config defaults.")
                norm_stats = None

        self.policy = _policy_config.create_trained_policy(
            config,
            ckpt_dir,
            norm_stats=norm_stats,
        )
        print(f"[Pi05LiberoModel] Loaded checkpoint from {ckpt_dir}")

        # Internal state.
        self.instruction: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_language(self, instruction: str) -> None:
        """Set the language instruction for the current episode."""
        if instruction != self.instruction:
            self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Run model inference and return an array of predicted actions.

        Parameters
        ----------
        agentview_image : np.ndarray, (H, W, 3), uint8
            Third-person (agent-view) camera image.
        wrist_image : np.ndarray, (H, W, 3), uint8
            Wrist (eye-in-hand) camera image.
        state : np.ndarray, (8,), float32
            Proprioceptive state: [eef_pos(3), axis_angle(3), gripper_qpos(2)].

        Returns
        -------
        actions : np.ndarray, (action_horizon, 7)
            Predicted actions (xyz + rpy + gripper deltas).
        """
        assert self.instruction is not None, "Call set_language() before predict()."

        # Build observation dict matching LIBERO policy input format.
        obs_dict = {
            "observation/image": preprocess_image(agentview_image),
            "observation/wrist_image": preprocess_image(wrist_image),
            "observation/state": state.astype(np.float32),
            "prompt": self.instruction,
        }

        raw_actions = self.policy.infer(obs_dict)["actions"]
        actions = np.asarray(raw_actions)[:self.action_horizon]
        return actions

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        """Convenience: predict directly from a LIBERO observation dict.

        Extracts images and state from the raw LIBERO ``obs`` dict.
        """
        agentview = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        """Reset internal state between episodes."""
        self.instruction = None
        print("[Pi05LiberoModel] Reset.")
