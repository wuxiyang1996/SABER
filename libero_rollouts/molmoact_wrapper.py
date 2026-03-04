"""MolmoAct wrapper for LIBERO.

Uses per-suite HuggingFace checkpoints from allenai/MolmoAct-7B-D-LIBERO-*.
Based on the Molmo vision-language model with action reasoning and spatial traces.

Inference follows the official README:
  generate() → model.parse_action(text, unnorm_key=...)

Checkpoints:
    allenai/MolmoAct-7B-D-LIBERO-{Spatial,Object,Goal,Long}-0812
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pi05_libero_model import build_libero_state, preprocess_image

_SUITE_TO_UNNORM_KEY = {
    "libero_spatial": "libero_spatial_no_noops_modified",
    "libero_object": "libero_object_no_noops_modified",
    "libero_goal": "libero_goal_no_noops_modified",
    "libero_10": "libero_10_no_noops_modified",
}


class MolmoActWrapper:
    def __init__(
        self,
        checkpoint: str = "allenai/MolmoAct-7B-D-LIBERO-Spatial-0812",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 1,
        replan_steps: int = 1,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.bfloat16
        self._unnorm_key = _SUITE_TO_UNNORM_KEY.get(suite_name, f"{suite_name}_no_noops_modified")

        print(f"[MolmoAct] Loading from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            padding_side="left",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        print(f"[MolmoAct] Ready on {device}.")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def _build_prompt(self, instruction: str) -> str:
        return (
            f"The task is {instruction}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {instruction}, "
            "let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
            "along with other images from different camera views as additional information, "
            "what is the action that the robot should take?"
        )

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import torch

        assert self.instruction is not None, "Call set_language() first."

        pil_agentview = Image.fromarray(agentview_image)
        pil_wrist = Image.fromarray(wrist_image)
        imgs = [pil_agentview, pil_wrist]

        prompt = self._build_prompt(self.instruction)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [dict(type="text", text=prompt)]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=[imgs],
            text=text,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=self._dtype):
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_tokens = generated_ids[:, inputs["input_ids"].size(1):]
        generated_text = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        action = self.model.parse_action(generated_text, unnorm_key=self._unnorm_key)
        if action and len(action) > 0:
            return np.array(action[0], dtype=np.float64).reshape(1, -1)[:, :7]
        return np.zeros((1, 7), dtype=np.float64)

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        agentview = preprocess_image(obs["agentview_image"])
        wrist = preprocess_image(obs["robot0_eye_in_hand_image"])
        state = build_libero_state(obs)
        return self.predict(agentview, wrist, state)

    def reset(self) -> None:
        self.instruction = None
