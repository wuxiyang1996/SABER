"""InternVLA-M1 wrapper for LIBERO.

Uses per-suite HuggingFace checkpoints from InternRobotics/InternVLA-M1-LIBERO-*.
Custom multi-module architecture (Qwen2.5VL + DINOv2 + QFormer + DiT diffusion).
Requires the InternVLA-M1 repo for model loading.

Action chunk: 8 actions per call
Action dim: 7

Checkpoints:
    InternRobotics/InternVLA-M1-LIBERO-{Spatial,Long}
    InternRobotics/InternVLA-M1  (base, for Object/Goal)
"""

from __future__ import annotations

import os
import sys
import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image


def _ensure_internvla_importable():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "repos", "internvla_m1"),
        os.path.join(os.path.dirname(__file__), "..", "repos", "InternVLA-M1"),
    ]
    for p in candidates:
        p = os.path.realpath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


def _resolve_checkpoint_pt(hf_repo_id: str) -> str:
    """Download HF repo and find the .pt checkpoint file."""
    from huggingface_hub import snapshot_download
    import glob

    local_dir = snapshot_download(repo_id=hf_repo_id)
    pt_files = glob.glob(os.path.join(local_dir, "checkpoints", "*.pt"))
    if not pt_files:
        pt_files = glob.glob(os.path.join(local_dir, "**", "*.pt"), recursive=True)
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt checkpoint found in {local_dir}. "
            f"Contents: {os.listdir(local_dir)}"
        )
    return sorted(pt_files)[-1]


class InternVLAWrapper:
    def __init__(
        self,
        checkpoint: str = "InternRobotics/InternVLA-M1-LIBERO-Spatial",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 8,
        replan_steps: int = 5,
    ):
        import torch

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16

        src_path = _ensure_internvla_importable()
        if src_path is None:
            raise RuntimeError(
                "InternVLA-M1 repo not found. Clone it: "
                "git clone https://github.com/InternRobotics/InternVLA-M1.git repos/internvla_m1"
            )

        from InternVLA.model.framework.M1 import InternVLA_M1
        from InternVLA.model.framework.share_tools import read_mode_config

        print(f"[InternVLA-M1] Resolving checkpoint for {checkpoint} ...")
        if os.path.isfile(checkpoint) and checkpoint.endswith(".pt"):
            ckpt_path = checkpoint
        else:
            ckpt_path = _resolve_checkpoint_pt(checkpoint)

        print(f"[InternVLA-M1] Loading from {ckpt_path} ...")
        self.model = InternVLA_M1.from_pretrained(ckpt_path)
        self.model = self.model.to(self._dtype).to(self.device).eval()

        _, norm_stats = read_mode_config(ckpt_path)
        unnorm_key = self._find_unnorm_key(norm_stats, suite_name)
        self._action_stats = norm_stats[unnorm_key]["action"]

        print(f"[InternVLA-M1] Ready on {device} (unnorm_key={unnorm_key}).")

    @staticmethod
    def _find_unnorm_key(norm_stats: dict, suite_name: str) -> str:
        if len(norm_stats) == 1:
            return next(iter(norm_stats.keys()))
        for key in norm_stats:
            if suite_name.replace("_", "") in key.replace("_", ""):
                return key
        return next(iter(norm_stats.keys()))

    @staticmethod
    def _unnormalize_actions(normalized: np.ndarray, stats: dict) -> np.ndarray:
        mask = np.array(stats.get("mask", [True] * 7), dtype=bool)
        hi = np.array(stats["max"], dtype=np.float64)
        lo = np.array(stats["min"], dtype=np.float64)
        actions = np.clip(normalized, -1, 1)
        actions[:, 6] = np.where(actions[:, 6] < 0.5, 0, 1)
        actions = np.where(mask, 0.5 * (actions + 1) * (hi - lo) + lo, actions)
        # Model outputs gripper in [0,1] (1=open, 0=close).
        # LIBERO expects [-1,+1] (-1=open, +1=close). Convert:
        actions[:, 6] = np.where(actions[:, 6] > 0.5, -1.0, 1.0)
        return actions

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        assert self.instruction is not None, "Call set_language() first."

        pil_agentview = Image.fromarray(agentview_image).convert("RGB")
        pil_wrist = Image.fromarray(wrist_image).convert("RGB")

        result = self.model.predict_action(
            batch_images=[[pil_agentview, pil_wrist]],
            instructions=[self.instruction],
            use_ddim=True,
            num_ddim_steps=10,
        )

        normalized = result["normalized_actions"][0]
        actions = self._unnormalize_actions(normalized, self._action_stats)
        return actions[:, :7].astype(np.float64)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image(obs["agentview_image"])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
