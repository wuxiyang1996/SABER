"""Fast DeepThinkVLA wrapper for LIBERO rollouts.

Optimised wrapper for rollout speed during cold-start collection and
GRPO training, while still capturing reasoning text (needed for
thinking_inflation and hallucination objectives).

Speed optimisations vs deepthinkvla_wrapper.py:
  - 4-bit NF4 quantization via bitsandbytes (~50% less VRAM, faster on A100)
  - SDPA attention
  - torch.inference_mode
  - Reuses preprocessed prompt across calls with same instruction
  - max_new_tokens=128 (CoT + action; truncates verbose reasoning)

Checkpoint: yinchenghust/deepthinkvla_libero_cot_rl
"""

from __future__ import annotations

import os
import re
import sys
import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image

_ACTION_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _ensure_deepthinkvla_importable():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "repos", "deepthinkvla"),
        os.path.join(os.path.dirname(__file__), "..", "repos", "DeepThinkVLA"),
        os.path.join(os.path.dirname(__file__), "..", ".cache", "repos", "deepthinkvla"),
    ]
    for p in candidates:
        p = os.path.realpath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


def _make_bnb_config():
    """Build a 4-bit NF4 quantization config if bitsandbytes is available."""
    try:
        import torch
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        return None


class DeepThinkVLARolloutWrapper:
    """Fast inference wrapper with reasoning capture and 4-bit quantization."""

    def __init__(
        self,
        checkpoint: str = "yinchenghust/deepthinkvla_libero_cot_rl",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 10,
        replan_steps: int = 5,
        max_new_tokens: int = 128,
        quantize_4bit: bool = True,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.max_new_tokens = max_new_tokens
        self.instruction = None
        self._dtype = torch.bfloat16
        self._cached_prompt = None
        self._last_reasoning = ""

        _ensure_deepthinkvla_importable()

        print(f"[DeepThinkVLA-rollout] Loading from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
        )

        bnb_config = _make_bnb_config() if quantize_4bit else None
        load_kwargs = dict(
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = {"": device}
            print(f"[DeepThinkVLA-rollout] Using 4-bit NF4 quantization.")
        else:
            print(f"[DeepThinkVLA-rollout] No quantization (bitsandbytes not available).")

        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, **load_kwargs,
        )
        if bnb_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[DeepThinkVLA-rollout] Ready on {device}  "
              f"(max_new_tokens={max_new_tokens}, "
              f"quantized={'4bit' if bnb_config else 'no'}).")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction
        self._cached_prompt = (
            instruction if "<image>" in instruction
            else "<image>" + instruction
        )

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import torch

        assert self._cached_prompt is not None, "Call set_language() first."

        pil_image = Image.fromarray(agentview_image)
        inputs = self.processor(
            text=self._cached_prompt, images=pil_image, return_tensors="pt",
        ).to(self.device, dtype=self._dtype)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        self._last_reasoning = decoded
        action = self._parse_action(decoded)

        single = np.array(action, dtype=np.float64).reshape(1, -1)[:, :7]
        if self.action_horizon > 1:
            return np.tile(single, (self.action_horizon, 1))
        return single

    @staticmethod
    def _parse_action(text: str) -> np.ndarray:
        nums = _ACTION_RE.findall(text)
        if len(nums) >= 7:
            return np.array([float(x) for x in nums[-7:]])
        return np.zeros(7)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image(obs["agentview_image"])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def get_last_reasoning(self) -> str:
        return self._last_reasoning

    def reset(self) -> None:
        self.instruction = None
        self._cached_prompt = None
        self._last_reasoning = ""
