"""GR00T N1.5 wrapper for LIBERO evaluation.

Uses per-suite fine-tuned Tacoin GR00T N1.5-3B checkpoints:
    Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL-8K
    Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT-8K
    Tacoin/GR00T-N1.5-3B-LIBERO-GOAL-8K
    Tacoin/GR00T-N1.5-3B-LIBERO-LONG-8K   (maps to libero_10)

Key details:
    - 3B parameter VLA (Eagle backbone + flow matching DiT action head)
    - Input: 2 images (224x224 after crop/resize) + EEF decomposed state + language
    - Output: per-component action dict, 16-step action chunks
    - Embodiment tag: new_embodiment (Tacoin fine-tuned)
    - State decomposition: x,y,z (EEF pos) + roll,pitch,yaw (axis-angle) + gripper (2D)
    - Action: x,y,z,roll,pitch,yaw (6D) + gripper (1D) = 7D per timestep
    - Uses Isaac-GR00T N1.5 API (repos/groot_n15)
    - Runs in vla_smolvla conda env (subprocess isolation)
"""

from __future__ import annotations

import math
import os
import sys
import types
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_GROOT_N15_REPO = os.path.join(os.path.dirname(_THIS_DIR), "repos", "groot_n15")
_GROOT_N15_LIBERO = os.path.join(_GROOT_N15_REPO, "examples", "Libero")

_IMAGE_SIZE = 256
_ACTION_HORIZON = 16


def _ensure_groot_n15_imports():
    """Set up sys.path and mock pytorch3d for Isaac-GR00T N1.5 imports."""
    if "pytorch3d" not in sys.modules:
        pt3d = types.ModuleType("pytorch3d")
        pt3d_transforms = types.ModuleType("pytorch3d.transforms")
        pt3d.transforms = pt3d_transforms
        sys.modules["pytorch3d"] = pt3d
        sys.modules["pytorch3d.transforms"] = pt3d_transforms

    if _GROOT_N15_REPO not in sys.path:
        sys.path.insert(0, _GROOT_N15_REPO)
    if _GROOT_N15_LIBERO not in sys.path:
        sys.path.insert(0, _GROOT_N15_LIBERO)


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle."""
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def _preprocess_libero_image(img: np.ndarray, size: int = _IMAGE_SIZE) -> np.ndarray:
    """Rotate 180 and resize for LIBERO convention."""
    from PIL import Image
    img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


class GR00TWrapper:
    def __init__(
        self,
        checkpoint: str = "Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL-8K",
        suite_name: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: int = _ACTION_HORIZON,
        replan_steps: int = 8,
    ):
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
        os.environ.setdefault("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")

        _ensure_groot_n15_imports()

        from gr00t.model.policy import Gr00tPolicy
        from custom_data_config import LiberoDataConfig

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None

        self.use_obs_predict = True

        data_config = LiberoDataConfig()
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        self._action_keys = data_config.action_keys

        print(f"[GR00T-N1.5] Loading from {checkpoint} on {device} ...", file=sys.stderr)
        self.policy = Gr00tPolicy(
            model_path=checkpoint,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=4,
            device=device,
        )
        print(f"[GR00T-N1.5] Ready. Action keys: {self._action_keys}", file=sys.stderr)

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def _build_observation(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        eef_pos: np.ndarray,
        eef_axisangle: np.ndarray,
        gripper_qpos: np.ndarray,
    ) -> dict:
        """Build N1.5 observation dict.

        Video:  (T=1, H, W, C) uint8
        State:  (T=1, D) float32 -- gripper is 1D (left finger only)
        Language: list[str] of length T=1
        """
        gripper_1d = np.array([[gripper_qpos[0]]], dtype=np.float32)
        obs = {
            "video.image": agentview_image[None],
            "video.wrist_image": wrist_image[None],
            "state.x": np.array([[eef_pos[0]]], dtype=np.float32),
            "state.y": np.array([[eef_pos[1]]], dtype=np.float32),
            "state.z": np.array([[eef_pos[2]]], dtype=np.float32),
            "state.roll": np.array([[eef_axisangle[0]]], dtype=np.float32),
            "state.pitch": np.array([[eef_axisangle[1]]], dtype=np.float32),
            "state.yaw": np.array([[eef_axisangle[2]]], dtype=np.float32),
            "state.gripper": gripper_1d,
            "annotation.human.action.task_description": [self.instruction],
        }
        return obs

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Predict from preprocessed 224/256 images + 8-dim state."""
        assert self.instruction is not None, "Call set_language() first."
        obs = self._build_observation(
            agentview_image, wrist_image,
            eef_pos=state[:3],
            eef_axisangle=state[3:6],
            gripper_qpos=state[6:8],
        )
        action_dict = self.policy.get_action(obs)
        return self._concat_actions(action_dict)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        """Predict from raw LIBERO observation dict."""
        assert self.instruction is not None, "Call set_language() first."

        agentview = _preprocess_libero_image(obs["agentview_image"])
        wrist = _preprocess_libero_image(obs["robot0_eye_in_hand_image"])
        eef_pos = obs["robot0_eef_pos"].astype(np.float32)
        eef_axisangle = _quat_to_axisangle(obs["robot0_eef_quat"]).astype(np.float32)
        gripper_qpos = obs["robot0_gripper_qpos"].astype(np.float32)

        groot_obs = self._build_observation(
            agentview, wrist, eef_pos, eef_axisangle, gripper_qpos,
        )
        action_dict = self.policy.get_action(groot_obs)
        return self._concat_actions(action_dict)

    def _concat_actions(self, action_dict: dict) -> np.ndarray:
        """Concatenate per-component actions into (T, 7) array.

        N1.5 get_action() denormalizes to the original training data space:
        - position/rotation: controller-space deltas (roughly [-1, 1])
        - gripper: [-1, 1] where -1=open, +1=close (matches LIBERO env)

        We just binarize the gripper for clean discrete control.
        """
        components = []
        for key in self._action_keys:
            arr = np.atleast_2d(action_dict[key])
            components.append(arr)
        actions = np.concatenate(components, axis=-1).astype(np.float64)

        actions[..., -1] = np.sign(actions[..., -1])

        return actions

    def reset(self) -> None:
        self.instruction = None
