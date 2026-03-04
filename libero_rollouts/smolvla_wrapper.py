"""SmolVLA wrapper for LIBERO.

Uses the HuggingFace LeRobot SmolVLA checkpoint:
    HuggingFaceVLA/smolvla_libero

Key details:
    - 450M parameter VLA (SmolVLM2-500M backbone)
    - Input: 2 images (256x256) + 8-dim state + language instruction
    - Output: 7-dim action (1 action per call, no chunking)
    - Normalization: MEAN_STD for state/action
    - Runs in vla_smolvla conda env
"""

from __future__ import annotations

import io
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pi05_libero_model import build_libero_state

_HF_REPO = "HuggingFaceVLA/smolvla_libero"
_IMAGE_SIZE = 256


def _preprocess_libero_image(img: np.ndarray, size: int = _IMAGE_SIZE) -> np.ndarray:
    """Rotate 180 and resize to target size for LIBERO convention."""
    img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


class SmolVLAWrapper:
    def __init__(
        self,
        checkpoint: str = _HF_REPO,
        suite_name: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: int = 1,
        replan_steps: int = 1,
    ):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None

        print(f"[SmolVLA] Loading from {checkpoint} ...")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy = self.policy.to(device).eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self._torch = torch
        print(f"[SmolVLA] Ready on {device}.")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        assert self.instruction is not None, "Call set_language() first."

        torch = self._torch

        # Convert images: (H,W,3) uint8 -> (1,3,H,W) float [0,1]
        img1 = torch.from_numpy(agentview_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2 = torch.from_numpy(wrist_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)

        batch = {
            "observation.images.image": img1,
            "observation.images.image2": img2,
            "observation.state": st,
            "task": self.instruction,
        }

        batch = self.preprocess(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocess(action)

        # action is (1, 7) tensor
        if isinstance(action, dict):
            action = action.get("action", action)
        action = action.cpu().float().numpy().astype(np.float64)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        return action

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = _preprocess_libero_image(obs["agentview_image"])
        wrist = _preprocess_libero_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
        self.policy.reset()
