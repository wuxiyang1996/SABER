"""GR00T N1.6 wrapper for LIBERO evaluation.

Uses the NVIDIA GR00T-N1.6-3B model via Isaac-GR00T library:
    nvidia/GR00T-N1.6-3B

Key details:
    - 3B parameter VLA (Eagle/Cosmos VLM backbone + flow matching DiT action head)
    - Input: 2 images (256x256) + decomposed state + language instruction
    - Output: per-component action dict, 16-step action chunks
    - Embodiment tag: LIBERO_PANDA (ID 2)
    - State decomposition: x,y,z (EEF pos) + roll,pitch,yaw (axis-angle) + gripper (2D)
    - Action: x,y,z,roll,pitch,yaw (6D) + gripper (1D) = 7D per timestep
    - Runs in vla_smolvla conda env with Isaac-GR00T on PYTHONPATH
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_GROOT_REPO = os.path.join(os.path.dirname(_THIS_DIR), "repos", "groot")
if _GROOT_REPO not in sys.path:
    sys.path.insert(0, _GROOT_REPO)

_HF_REPO = "nvidia/GR00T-N1.6-3B"
_IMAGE_SIZE = 256
_ACTION_HORIZON = 16


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


_LIBERO_MODALITY_CONFIG = {
    "video": {
        "delta_indices": [0],
        "modality_keys": ["image", "wrist_image"],
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": None,
    },
    "state": {
        "delta_indices": [0],
        "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": None,
    },
    "action": {
        "delta_indices": list(range(16)),
        "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": [
            {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            for _ in range(7)
        ],
    },
    "language": {
        "delta_indices": [0],
        "modality_keys": ["annotation.human.action.task_description"],
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": None,
    },
}


def _ensure_libero_modality_config(checkpoint: str) -> None:
    """Inject libero_panda modality config into the model's cached processor_config.json.

    The base nvidia/GR00T-N1.6-3B model ships without libero_panda configs.
    We inject them from the test fixtures bundled with Isaac-GR00T.
    """
    import json
    from huggingface_hub import try_to_load_from_cache

    pc_path = try_to_load_from_cache(checkpoint, "processor_config.json")
    if pc_path is None or isinstance(pc_path, str) and not os.path.isfile(pc_path):
        return

    with open(pc_path) as f:
        pc = json.load(f)

    mc = pc.get("processor_kwargs", {}).get("modality_configs", {})
    if "libero_panda" in mc:
        return

    mc["libero_panda"] = _LIBERO_MODALITY_CONFIG
    with open(pc_path, "w") as f:
        json.dump(pc, f, indent=2)
    print("[GR00T] Injected libero_panda modality config into processor_config.json")

    # Also inject into statistics.json and embodiment_id.json
    for fname, key, default in [
        ("embodiment_id.json", "libero_panda", 2),
    ]:
        fpath = try_to_load_from_cache(checkpoint, fname)
        if fpath and os.path.isfile(fpath):
            with open(fpath) as f:
                data = json.load(f)
            if "libero_panda" not in data:
                data["libero_panda"] = default
                with open(fpath, "w") as f:
                    json.dump(data, f, indent=2)

    stats_path = try_to_load_from_cache(checkpoint, "statistics.json")
    if stats_path and os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if "libero_panda" not in stats:
            fixture_stats = os.path.join(
                _GROOT_REPO, "tests", "fixtures", "processor_config", "statistics.json"
            )
            if os.path.isfile(fixture_stats):
                with open(fixture_stats) as f:
                    fix = json.load(f)
                if "libero_panda" in fix:
                    stats["libero_panda"] = fix["libero_panda"]
                    with open(stats_path, "w") as f:
                        json.dump(stats, f, indent=2)


class GR00TWrapper:
    def __init__(
        self,
        checkpoint: str = _HF_REPO,
        suite_name: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: int = _ACTION_HORIZON,
        replan_steps: int = 5,
    ):
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
        os.environ.setdefault("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None

        self.use_obs_predict = True

        _ensure_libero_modality_config(checkpoint)

        print(f"[GR00T] Loading from {checkpoint} on {device} ...")
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.LIBERO_PANDA,
            model_path=checkpoint,
            device=device,
        )
        self._modality_configs = self.policy.get_modality_config()
        self._action_keys = self._modality_configs["action"].modality_keys
        self._language_key = self._modality_configs["language"].modality_keys[0]
        print(f"[GR00T] Ready. Action keys: {self._action_keys}, "
              f"Language key: {self._language_key}")

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
        """Build the observation dict expected by Gr00tPolicy.get_action().

        Video:  dict[key] -> np.ndarray (B=1, T=1, H, W, C) uint8
        State:  dict[key] -> np.ndarray (B=1, T=1, D) float32
        Language: dict[key] -> list[list[str]] shape (B=1, 1)
        """
        obs = {
            "video": {
                "image": agentview_image[None, None],       # (1,1,H,W,3)
                "wrist_image": wrist_image[None, None],     # (1,1,H,W,3)
            },
            "state": {
                "x": np.array([[[eef_pos[0]]]], dtype=np.float32),
                "y": np.array([[[eef_pos[1]]]], dtype=np.float32),
                "z": np.array([[[eef_pos[2]]]], dtype=np.float32),
                "roll": np.array([[[eef_axisangle[0]]]], dtype=np.float32),
                "pitch": np.array([[[eef_axisangle[1]]]], dtype=np.float32),
                "yaw": np.array([[[eef_axisangle[2]]]], dtype=np.float32),
                "gripper": gripper_qpos.astype(np.float32)[None, None],  # (1,1,2)
            },
            "language": {
                self._language_key: [[self.instruction]],
            },
        }
        return obs

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """Predict actions from preprocessed observations.

        The `state` array is the 8-dim LIBERO state built by build_libero_state():
        [joint_pos(6?), gripper_qpos(2)] -- but for GR00T we need EEF decomposition.

        NOTE: This method is less ideal for GR00T because the standard
        predict() interface doesn't include EEF pose. Use predict_from_obs()
        for best results.
        """
        assert self.instruction is not None, "Call set_language() first."
        obs = self._build_observation(
            agentview_image, wrist_image,
            eef_pos=state[:3],
            eef_axisangle=state[3:6],
            gripper_qpos=state[6:8],
        )
        action_dict, _ = self.policy.get_action(obs)
        return self._concat_actions(action_dict)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        """Predict actions directly from raw LIBERO observation dict."""
        assert self.instruction is not None, "Call set_language() first."

        agentview = _preprocess_libero_image(obs["agentview_image"])
        wrist = _preprocess_libero_image(obs["robot0_eye_in_hand_image"])
        eef_pos = obs["robot0_eef_pos"].astype(np.float32)
        eef_axisangle = _quat_to_axisangle(obs["robot0_eef_quat"]).astype(np.float32)
        gripper_qpos = obs["robot0_gripper_qpos"].astype(np.float32)

        groot_obs = self._build_observation(
            agentview, wrist, eef_pos, eef_axisangle, gripper_qpos,
        )
        action_dict, _ = self.policy.get_action(groot_obs)
        return self._concat_actions(action_dict)

    def _concat_actions(self, action_dict: dict) -> np.ndarray:
        """Concatenate per-component actions into (T, 7) array.

        GR00T returns: dict[key] -> (B=1, T=16, D) float32
        We need: (T, 7) float64

        Matches the official Isaac-GR00T LiberoEnv.step() post-processing:
        gripper is normalized from [0,1] to [-1,+1] then sign-inverted so that
        the raw LIBERO env receives +1=close, -1=open.
        """
        components = []
        for key in self._action_keys:
            arr = action_dict[key]            # (B, T, D)
            components.append(arr[0])         # (T, D)
        actions = np.concatenate(components, axis=-1)  # (T, 7)
        actions = actions.astype(np.float64)

        # Gripper post-processing (matches official libero_env.py)
        actions[..., -1] = 2.0 * actions[..., -1] - 1.0   # [0,1] -> [-1,+1]
        actions[..., -1] = np.sign(actions[..., -1])       # binarize
        actions[..., -1] = -actions[..., -1]               # invert

        return actions

    def reset(self) -> None:
        self.instruction = None
        self.policy.reset()
