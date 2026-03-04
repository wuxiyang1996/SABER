"""OpenVLA-OFT wrapper for LIBERO.

Uses the combined multi-suite checkpoint from HuggingFace:
    moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10

Key differences from standard OpenVLA:
    - Continuous action output via L1 regression MLP head (not discrete tokens)
    - Proprioceptive state projected into LLM embedding space
    - Two images: agentview + wrist camera
    - Action chunking: 8 actions per inference call
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pi05_libero_model import build_libero_state

_SUITE_TO_UNNORM_KEY = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
_IMAGE_SIZE = 224
_HF_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"


def _preprocess_image(img: np.ndarray, size: int = _IMAGE_SIZE) -> np.ndarray:
    """Rotate 180, JPEG round-trip, Lanczos resize — matches RLDS pipeline."""
    img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    pil = Image.open(buf)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def _center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    w, h = image.size
    new_w = int(w * (crop_scale ** 0.5))
    new_h = int(h * (crop_scale ** 0.5))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h)).resize(
        (_IMAGE_SIZE, _IMAGE_SIZE), Image.LANCZOS
    )


def _load_component_state_dict(path: str) -> dict:
    """Load checkpoint and strip DDP 'module.' prefix if present."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _normalize_proprio(proprio: np.ndarray, stats: dict) -> np.ndarray:
    mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high = np.array(stats["q99"])
    low = np.array(stats["q01"])
    return np.clip(
        np.where(mask, 2 * (proprio - low) / (high - low + 1e-8) - 1, proprio),
        -1.0, 1.0,
    )


class OpenVLAOFTWrapper:
    def __init__(
        self,
        checkpoint: str = _HF_REPO,
        suite_name: str | None = None,
        device: str = "cuda:0",
        action_horizon: int = NUM_ACTIONS_CHUNK,
        replan_steps: int = 5,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from huggingface_hub import hf_hub_download
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16

        self._unnorm_key = _SUITE_TO_UNNORM_KEY.get(suite_name, "libero_spatial_no_noops")

        print(f"[OpenVLA-OFT] Loading processor from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

        print(f"[OpenVLA-OFT] Loading model from {checkpoint} ...")
        import timm
        _real_timm_ver = timm.__version__
        timm.__version__ = "0.9.16"
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
        finally:
            timm.__version__ = _real_timm_ver

        from openvla_wrapper import _patch_prismatic_vision_backbone
        _patch_prismatic_vision_backbone(self.model)
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.eval()

        # Load dataset statistics for action/proprio normalization
        stats_path = hf_hub_download(repo_id=checkpoint, filename="dataset_statistics.json")
        with open(stats_path) as f:
            self.model.norm_stats = json.load(f)

        llm_dim = self.model.llm_dim

        # Action head (L1 regression MLP)
        print("[OpenVLA-OFT] Loading action head ...")
        self.action_head = L1RegressionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM,
        ).to(self._dtype).to(self.device)
        self.action_head.eval()
        ah_path = hf_hub_download(
            repo_id=checkpoint, filename="action_head--300000_checkpoint.pt",
        )
        self.action_head.load_state_dict(_load_component_state_dict(ah_path))

        # Proprio projector
        print("[OpenVLA-OFT] Loading proprio projector ...")
        self.proprio_projector = ProprioProjector(
            llm_dim=llm_dim, proprio_dim=PROPRIO_DIM,
        ).to(self._dtype).to(self.device)
        self.proprio_projector.eval()
        pp_path = hf_hub_download(
            repo_id=checkpoint, filename="proprio_projector--300000_checkpoint.pt",
        )
        self.proprio_projector.load_state_dict(_load_component_state_dict(pp_path))

        print(f"[OpenVLA-OFT] Ready on {device}  (unnorm_key={self._unnorm_key}).")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def set_suite(self, suite_name: str) -> None:
        """Switch unnorm key when suite changes (single checkpoint serves all)."""
        self.suite_name = suite_name
        self._unnorm_key = _SUITE_TO_UNNORM_KEY.get(suite_name, "libero_spatial_no_noops")

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import torch

        assert self.instruction is not None, "Call set_language() first."

        # Preprocess both images
        primary_pil = _center_crop(Image.fromarray(agentview_image))
        wrist_pil = _center_crop(Image.fromarray(wrist_image))

        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, primary_pil).to(self.device, dtype=self._dtype)

        # Concatenate wrist image pixel values
        wrist_inputs = self.processor(prompt, wrist_pil).to(self.device, dtype=self._dtype)
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1,
        )

        # Normalize proprio
        proprio_stats = self.model.norm_stats[self._unnorm_key]["proprio"]
        norm_proprio = _normalize_proprio(state, proprio_stats)

        with torch.inference_mode():
            actions, _ = self.model.predict_action(
                **inputs,
                unnorm_key=self._unnorm_key,
                do_sample=False,
                proprio=norm_proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
            )

        # actions shape: (NUM_ACTIONS_CHUNK, ACTION_DIM) — numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().float().numpy()

        actions = np.array(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # Model outputs gripper in [0,1] (1=open, 0=close; mask=False → not
        # unnormalized).  LIBERO expects [-1,+1] (-1=open, +1=close). Convert:
        actions[:, 6] = np.where(actions[:, 6] > 0.5, -1.0, 1.0)

        return actions

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = _preprocess_image(obs["agentview_image"])
        wrist = _preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
