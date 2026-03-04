"""OpenVLA wrapper for LIBERO.

Uses HuggingFace transformers with per-suite finetuned checkpoints.
Each call to predict() returns a single 7-dim action (no chunking).

Checkpoints:
    openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state


def preprocess_image_openvla(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess LIBERO image for OpenVLA, matching the RLDS training pipeline.

    Steps match the official OpenVLA eval (get_libero_image + resize_image):
      1. Rotate 180 degrees (LIBERO convention)
      2. JPEG encode/decode (matches RLDS dataset builder compression)
      3. Lanczos3 resize to target size
    """
    img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


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



def _center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    """Take center crop with `crop_scale` fraction of the original area.

    OpenVLA LIBERO finetuning uses random 90% area crops during training, so
    at test time we take the center 90% crop (matching official eval scripts).
    """
    w, h = image.size
    new_w = int(w * (crop_scale ** 0.5))
    new_h = int(h * (crop_scale ** 0.5))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


class OpenVLAWrapper:
    def __init__(
        self,
        checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 1,
        replan_steps: int = 1,
        center_crop: bool = True,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16
        self._center_crop = center_crop

        # unnorm_key matches LIBERO suite names used during OpenVLA finetuning
        self._unnorm_key = suite_name

        print(f"[OpenVLA] Loading processor from openvla/openvla-7b ...")
        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True,
        )

        print(f"[OpenVLA] Loading model from {checkpoint} ...")
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
        print(f"[OpenVLA] Ready on {device}.")

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
        if self._center_crop:
            pil_image = _center_crop(pil_image)

        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=self._dtype)

        action_dim = self.model.get_action_dim(self._unnorm_key)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=action_dim,
                pixel_values=inputs["pixel_values"],
                do_sample=False,
            )

        predicted_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
        discretized = self.model.vocab_size - predicted_token_ids
        discretized = np.clip(discretized - 1, 0, self.model.bin_centers.shape[0] - 1)
        normalized = self.model.bin_centers[discretized]

        stats = self.model.get_action_stats(self._unnorm_key)
        mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
        q99 = np.array(stats["q99"])
        q01 = np.array(stats["q01"])
        action = np.where(
            mask,
            0.5 * (normalized + 1) * (q99 - q01) + q01,
            normalized,
        )

        action = action.astype(np.float64)

        # Gripper post-processing: raw bin center is in [-1,1] (mask=False).
        # Positive values → open, negative values → close (model convention).
        # LIBERO env: -1=open, +1=close. Binarize and invert.
        action[-1] = -np.sign(action[-1]) if action[-1] != 0 else -1.0

        return action.reshape(1, -1)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image_openvla(obs["agentview_image"])
        wrist = preprocess_image_openvla(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
