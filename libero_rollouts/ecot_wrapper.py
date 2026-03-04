"""Embodied Chain-of-Thought (ECoT) wrapper for LIBERO.

ECoT extends OpenVLA with chain-of-thought reasoning before action prediction.
Uses the Embodied-CoT/ecot-openvla-7b-bridge checkpoint with OpenVLA's
transformers-based inference. Each predict() call returns a single 7-dim action.

Reference: https://github.com/MichalZawalski/embodied-CoT
Checkpoint: Embodied-CoT/ecot-openvla-7b-bridge
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image


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


class ECoTWrapper:
    def __init__(
        self,
        checkpoint: str = "Embodied-CoT/ecot-openvla-7b-bridge",
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
        self._last_reasoning = ""

        print(f"[ECoT] Loading processor from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
        )

        print(f"[ECoT] Loading model from {checkpoint} ...")
        import timm
        _real_timm_ver = timm.__version__
        timm.__version__ = "0.9.16"  # bypass OpenVLA's strict version gate
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
        print(f"[ECoT] Ready on {device}.")

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

    def get_last_reasoning(self) -> str:
        return self._last_reasoning

    def reset(self) -> None:
        self.instruction = None
        self._last_reasoning = ""
