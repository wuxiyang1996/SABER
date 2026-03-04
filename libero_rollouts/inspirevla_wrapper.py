"""MiniVLA / InspireVLA wrapper for LIBERO.

Uses the openvla-mini (Prismatic) inference stack to load and run MiniVLA-family
models. The model is loaded via ``prismatic.models.load.load_vla()`` which handles
the Prismatic checkpoint format (.pt), VQ/non-VQ action tokenizers, and
dataset statistics for action un-normalization.

Supported checkpoints:
  - Stanford-ILIAD/minivla-libero90-prismatic  (non-VQ, extra_action_tokenizer)
  - InspireVLA/minivla-inspire-libero-union4    (VQ, requires VQ-VAE model)

The VQ checkpoint requires a pre-trained VQ-VAE model at a specific path.
If the VQ-VAE is not available, the wrapper falls back to the non-VQ model.

Both ``minivla`` and ``inspirevla`` model IDs use this wrapper; the only
difference is the default checkpoint passed in.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pi05_libero_model import build_libero_state

_FRAMEWORK_DIR = os.path.dirname(_THIS_DIR)
_OPENVLA_MINI_DIR = os.path.join(_FRAMEWORK_DIR, "repos", "openvla_mini")

_HF_REPO_VQ = "InspireVLA/minivla-inspire-libero-union4"
_HF_REPO_NONVQ = "Stanford-ILIAD/minivla-libero90-prismatic"
_IMAGE_SIZE = 224

_NONVQ_UNION4_STATS = None


def _preprocess_libero_image(img: np.ndarray, size: int = _IMAGE_SIZE) -> np.ndarray:
    """Vertical flip (LIBERO convention for MiniVLA) and resize to target size.

    The openvla-mini eval uses ``np.flipud`` (vertical flip only), NOT a full
    180-degree rotation.  Using ``[::-1, ::-1]`` would left-right mirror the
    image vs. what the model saw at training time, causing terrible performance.
    """
    img = np.ascontiguousarray(np.flipud(img))
    pil = Image.fromarray(img)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def _ensure_prismatic_on_path():
    """Add openvla-mini repo to front of sys.path so the full prismatic
    package shadows any partial prismatic trees elsewhere (e.g. libero_rollouts/)."""
    # Must be first entry to take precedence over partial prismatic packages
    if not sys.path or sys.path[0] != _OPENVLA_MINI_DIR:
        while _OPENVLA_MINI_DIR in sys.path:
            sys.path.remove(_OPENVLA_MINI_DIR)
        sys.path.insert(0, _OPENVLA_MINI_DIR)
    os.environ.setdefault("PRISMATIC_DATA_ROOT", "/tmp")


def _download_hf_snapshot(repo_id: str) -> str:
    """Download a HuggingFace repo snapshot and return the local path.

    Only downloads the best (highest step) checkpoint to conserve disk space.
    """
    from huggingface_hub import list_repo_files, snapshot_download

    # Identify the best checkpoint so we can skip the others
    all_files = list_repo_files(repo_id)
    ckpt_files = sorted(
        [f for f in all_files if f.startswith("checkpoints/") and f.endswith(".pt")]
    )
    ignore = []
    if len(ckpt_files) > 1:
        # Keep only the last checkpoint (highest step number, sorted lexically)
        ignore = ckpt_files[:-1]

    return snapshot_download(repo_id=repo_id, ignore_patterns=ignore)


def _find_checkpoint(run_dir: str) -> Optional[str]:
    """Find the best .pt checkpoint in run_dir/checkpoints/."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    pts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and os.path.exists(os.path.join(ckpt_dir, f))],
    )
    if not pts:
        return None
    return os.path.join(ckpt_dir, pts[-1])


def _fix_config_base_vlm(run_dir: str) -> None:
    """Fix the base_vlm path in config.json to match the Prismatic registry key."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)

    vla = config.get("vla", {})
    base_vlm = vla.get("base_vlm", "")

    # The HF configs use training-time paths like
    # "pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b".
    # The Prismatic registry key is "prism-qwen25-extra-dinosiglip-224px+0_5b".
    if not os.path.isdir(base_vlm):
        fixed = base_vlm.split("/")[-1]
        # Replace the last dash before "0_5b" with "+": "-0_5b" → "+0_5b"
        if "-0_5b" in fixed:
            fixed = fixed.replace("-0_5b", "+0_5b", 1)
        if fixed != base_vlm:
            vla["base_vlm"] = fixed
            config["vla"] = vla
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)


def _vq_vae_available() -> bool:
    """Check if the VQ-VAE model files exist relative to openvla-mini repo."""
    vq_dir = os.path.join(
        _OPENVLA_MINI_DIR,
        "vq", "pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512",
    )
    model_pt = os.path.join(vq_dir, "checkpoints", "model.pt")
    config_json = os.path.join(vq_dir, "config.json")
    return os.path.isfile(model_pt) and os.path.isfile(config_json)


def _get_union4_dataset_stats() -> dict:
    """Download and cache the InspireVLA union4 dataset statistics."""
    global _NONVQ_UNION4_STATS
    if _NONVQ_UNION4_STATS is not None:
        return _NONVQ_UNION4_STATS
    try:
        from huggingface_hub import hf_hub_download
        stats_path = hf_hub_download(
            repo_id=_HF_REPO_VQ, filename="dataset_statistics.json"
        )
        with open(stats_path) as f:
            _NONVQ_UNION4_STATS = json.load(f)
        return _NONVQ_UNION4_STATS
    except Exception:
        return {}


class InspireVLAWrapper:
    def __init__(
        self,
        checkpoint: str = _HF_REPO_VQ,
        suite_name: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: int = 1,
        replan_steps: int = 1,
    ):
        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._model = None
        self._unnorm_key = None
        self._use_vq = False

        _ensure_prismatic_on_path()

        use_vq = (checkpoint == _HF_REPO_VQ) and _vq_vae_available()
        # Only fall back to NONVQ when the VQ checkpoint was requested but VAE is missing
        if checkpoint == _HF_REPO_VQ and not use_vq:
            repo_id = _HF_REPO_NONVQ
            print(
                f"[MiniVLA] VQ-VAE model not found at {_OPENVLA_MINI_DIR}/vq/. "
                f"Falling back to non-VQ model: {_HF_REPO_NONVQ}"
            )
        else:
            repo_id = checkpoint

        print(f"[MiniVLA] Downloading model snapshot: {repo_id} ...")
        run_dir = _download_hf_snapshot(repo_id)

        _fix_config_base_vlm(run_dir)

        ckpt_path = _find_checkpoint(run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No .pt checkpoint found in {run_dir}/checkpoints/"
            )

        if use_vq:
            self._unnorm_key = suite_name or "libero_spatial"
        else:
            with open(os.path.join(run_dir, "config.json")) as f:
                cfg = json.load(f)
            data_mix = cfg.get("vla", {}).get("data_mix", "libero_90")
            self._unnorm_key = data_mix

        # load_vla must be called from openvla-mini root for VQ-VAE relative paths
        orig_cwd = os.getcwd()
        try:
            os.chdir(_OPENVLA_MINI_DIR)

            import torch
            torch.set_num_threads(2)
            from prismatic.models.load import load_vla

            print(f"[MiniVLA] Loading VLA from {ckpt_path} ...")
            self._model = load_vla(
                ckpt_path,
                hf_token=os.environ.get("HF_TOKEN", None),
                load_for_training=False,
            )

            self._model.vision_backbone.to(
                dtype=self._model.vision_backbone.half_precision_dtype
            )
            self._model.llm_backbone.to(
                dtype=self._model.llm_backbone.half_precision_dtype
            )
            self._model.to(dtype=self._model.llm_backbone.half_precision_dtype)
            self._model.to(device)
            self._model.eval()

            self._use_vq = use_vq
        finally:
            os.chdir(orig_cwd)

        # For InspireVLA fallback (VQ checkpoint requested but VQ-VAE missing):
        # inject per-suite stats from the union4 repo so unnormalization uses
        # the correct per-suite q01/q99.
        if not use_vq and checkpoint == _HF_REPO_VQ:
            union4_stats = _get_union4_dataset_stats()
            if suite_name and suite_name in union4_stats:
                self._model.norm_stats[suite_name] = union4_stats[suite_name]
                self._unnorm_key = suite_name

        print(f"[MiniVLA] Ready on {device} (VQ={self._use_vq}, "
              f"unnorm_key={self._unnorm_key}).")

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

        with torch.inference_mode():
            action = self._model.predict_action(
                pil_image,
                self.instruction,
                unnorm_key=self._unnorm_key,
                do_sample=False,
            )

        action = action.astype(np.float64)

        # Gripper: predict_action() returns [0,1] (mask=False in dataset stats).
        # LIBERO env expects [-1,+1] where -1=open, +1=close.
        # Match openvla-mini eval: [0,1] → binarize at 0.5 → invert sign.
        action[-1] = 1.0 if action[-1] < 0.5 else -1.0

        return action.reshape(1, -1)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = _preprocess_libero_image(obs["agentview_image"])
        wrist = _preprocess_libero_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
