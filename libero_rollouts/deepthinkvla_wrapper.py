"""DeepThinkVLA wrapper for LIBERO.

Chain-of-thought + RL reasoning VLA based on PaliGemma. Uses the custom
DeepThinkVLA model class from the DeepThinkVLA repo for loading and inference.

Architecture: PaliGemma + hybrid attention (CoT autoregressive + actions parallel)
Action chunk: 10 actions per call (same as Pi0/Pi0.5)
Action dim: 7

Checkpoint: yinchenghust/deepthinkvla_libero_cot_rl
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image


def _ensure_deepthinkvla_importable():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "repos", "deepthinkvla", "src"),
        os.path.join(os.path.dirname(__file__), "..", "repos", "DeepThinkVLA", "src"),
    ]
    for p in candidates:
        p = os.path.realpath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None


class DeepThinkVLAWrapper:
    def __init__(
        self,
        checkpoint: str = "yinchenghust/deepthinkvla_libero_cot_rl",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 10,
        replan_steps: int = 5,
    ):
        import torch

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16
        self._last_reasoning = ""

        src_path = _ensure_deepthinkvla_importable()
        if src_path is None:
            raise RuntimeError(
                "DeepThinkVLA repo not found. Clone it: "
                "git clone https://github.com/OpenBMB/DeepThinkVLA.git repos/deepthinkvla"
            )

        from sft.modeling_deepthinkvla import DeepThinkVLA
        from dt_datasets.normalize import Unnormalize_Action
        from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK
        from transformers import AutoProcessor, GenerationConfig

        print(f"[DeepThinkVLA] Loading from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(checkpoint)

        self.model = DeepThinkVLA.from_pretrained(
            checkpoint,
            torch_dtype=self._dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        norm_stats_path = os.path.join(checkpoint, "norm_stats.json")
        if not os.path.isfile(norm_stats_path):
            from huggingface_hub import hf_hub_download
            norm_stats_path = hf_hub_download(repo_id=checkpoint, filename="norm_stats.json")

        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)
        for key in norm_stats["action"]:
            norm_stats["action"][key] = np.array(norm_stats["action"][key], dtype=np.float64)

        self._unnormalize = Unnormalize_Action(
            normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
            stats=norm_stats["action"],
            action_mask=ACTION_MASK,
        )

        self._gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=None,
            use_cache=True,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        self._think_prefix = (
            "First output the thinking process in <think></think> tags "
            "and then output the final action in <action></action>."
        )
        self._image_token = self.processor.tokenizer.additional_special_tokens[0]
        self._num_images = 2
        print(f"[DeepThinkVLA] Ready on {device}.")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import torch
        from torchvision import transforms

        assert self.instruction is not None, "Call set_language() first."

        resize = transforms.Resize((224, 224))
        pil_agentview = resize(Image.fromarray(agentview_image).convert("RGB"))
        pil_wrist = resize(Image.fromarray(wrist_image).convert("RGB"))
        images = [pil_agentview, pil_wrist]

        prompt = (
            self._image_token * self._num_images
            + self._think_prefix
            + f"Task: {self.instruction.lower()};"
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt",
        ).to(self.device, dtype=self._dtype)

        with torch.inference_mode():
            normalized_actions, input_cot_ids = self.model.predict_cot_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                generation_config=self._gen_config,
            )
            actions = self._unnormalize(
                torch.from_numpy(normalized_actions)
            ).numpy()

            self._last_reasoning = self.processor.tokenizer.decode(
                input_cot_ids[0, inputs["input_ids"].shape[-1]:-1]
            )

        return actions[:, :7].astype(np.float64)

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
