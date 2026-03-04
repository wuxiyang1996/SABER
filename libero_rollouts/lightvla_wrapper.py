"""LightVLA wrapper for LIBERO.

Uses per-suite HuggingFace checkpoints from TTJiang/LightVLA-libero-*.
Requires the LightVLA repo (https://github.com/LiAutoAD/LightVLA) to be
installed or cloned into repos/lightvla.

Checkpoints:
    TTJiang/LightVLA-libero-{spatial,object,goal,10}
"""

from __future__ import annotations

import os
import sys
import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image


def _ensure_lightvla_importable():
    """Add LightVLA repo to sys.path if available."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "repos", "lightvla"),
        os.path.join(os.path.dirname(__file__), "..", ".cache", "repos", "lightvla"),
    ]
    for p in candidates:
        p = os.path.realpath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


def _patch_prismatic_vision_backbone(model):
    """Fix timm >= 1.0 compat: get_intermediate_layers returns list, not tuple."""
    vb = getattr(model, "vision_backbone", None)
    if vb is None:
        return
    for attr in ("featurizer", "fused_featurizer"):
        feat = getattr(vb, attr, None)
        if feat is None:
            continue
        orig_fwd = feat.forward
        def _patched(x, _orig=orig_fwd):
            result = _orig(x)
            return result[0] if isinstance(result, (tuple, list)) else result
        feat.forward = _patched


class LightVLAWrapper:
    def __init__(
        self,
        checkpoint: str = "TTJiang/LightVLA-libero-spatial",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 1,
        replan_steps: int = 1,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16
        self._unnorm_key = suite_name

        _ensure_lightvla_importable()

        print(f"[LightVLA] Loading processor from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
        )

        print(f"[LightVLA] Loading model from {checkpoint} ...")
        import timm
        _real_timm_ver = timm.__version__
        timm.__version__ = "0.9.16"
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)
        finally:
            timm.__version__ = _real_timm_ver
        _patch_prismatic_vision_backbone(self.model)
        self.model.eval()
        print(f"[LightVLA] Ready on {device}.")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import torch

        assert self.instruction is not None, "Call set_language() first."

        pil_image = Image.fromarray(agentview_image)
        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=self._dtype)

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self._unnorm_key, do_sample=False,
            )

        return np.array(action, dtype=np.float64).reshape(1, -1)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image(obs["agentview_image"])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
