"""X-VLA wrapper for LIBERO.

Uses the unified X-VLA checkpoint (2toINF/X-VLA-Libero) for all LIBERO suites.
X-VLA uses absolute end-effector actions (not delta), so the LIBERO controller
must be switched to absolute mode (use_delta=False) before rollouts.

Architecture: Florence2 encoder + SoftPromptedTransformer action head.
Action format: 20D EE6D (pos3 + rot6d6 + grip1 + padding10), chunked (30 actions).
Proprio format: 20D (pos3 + rot6d6 + grip1 + zeros10).

Checkpoint:
    2toINF/X-VLA-Libero  (all suites, single checkpoint)
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


def _mat_to_rotate6d(rot_mat: np.ndarray) -> np.ndarray:
    """Extract 6D rotation from a 3x3 rotation matrix (first two columns)."""
    return np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]])


def _rotate6d_to_axisangle(rot6d: np.ndarray) -> np.ndarray:
    """Convert (N, 6) or (6,) 6D rotation to axis-angle (N, 3) or (3,).

    Uses Gram-Schmidt orthogonalisation on columns, then scipy for
    rotation matrix -> rotation vector (axis * angle).
    """
    single = rot6d.ndim == 1
    if single:
        rot6d = rot6d[np.newaxis]

    a1 = rot6d[:, :3]
    a2 = rot6d[:, 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    rot_mats = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)
    aa = R.from_matrix(rot_mats).as_rotvec()  # (N, 3)
    return aa[0] if single else aa


class XVLAWrapper:
    """X-VLA wrapper for LIBERO evaluation.

    Important: this model outputs **absolute** end-effector actions.
    The calling code must set ``robot.controller.use_delta = False``
    on the LIBERO env before running rollouts.

    Attributes
    ----------
    uses_absolute_actions : bool
        Signals that actions are absolute targets, not deltas.
    use_obs_predict : bool
        Signals that ``predict_from_obs`` should be called instead
        of ``predict``, because X-VLA needs raw EE pose from the obs.
    """

    uses_absolute_actions = True
    use_obs_predict = True

    DOMAIN_ID = 3        # LIBERO / Franka domain in X-VLA
    DENOISE_STEPS = 10   # flow-matching denoising iterations

    def __init__(
        self,
        checkpoint: str = "2toINF/X-VLA-Libero",
        suite_name: str = "libero_spatial",
        device: str = "cuda:0",
        action_horizon: int = 30,
        replan_steps: int = 5,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction = None
        self._dtype = torch.float32

        print(f"[X-VLA] Loading processor from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True,
        )

        print(f"[X-VLA] Loading model from {checkpoint} ...")
        self.model = AutoModel.from_pretrained(
            checkpoint, trust_remote_code=True, torch_dtype=self._dtype,
        ).to(self.device).eval()
        print(f"[X-VLA] Ready on {device}.")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        """Full inference from a raw LIBERO observation dict.

        Returns absolute 7D actions: [target_pos(3), target_axisangle(3), grip(1)]
        """
        import torch

        assert self.instruction is not None, "Call set_language() first."

        # --- images ---
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = obs["robot0_eye_in_hand_image"]
        images = [Image.fromarray(agentview), Image.fromarray(wrist)]

        # --- proprioception: [pos3, rot6d6, grip1, zeros10] = 20D ---
        # Use controller's site-frame EE pose when injected by the policy fn,
        # because obs robot0_eef_quat uses the MuJoCo body frame which differs.
        if "_ctrl_ee_ori_mat" in obs:
            ee_pos = np.asarray(obs["_ctrl_ee_pos"], dtype=np.float32)
            rot6d = _mat_to_rotate6d(
                np.asarray(obs["_ctrl_ee_ori_mat"], dtype=np.float64)
            ).astype(np.float32)
        else:
            ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            rot_mat = R.from_quat(ee_quat).as_matrix()
            rot6d = _mat_to_rotate6d(rot_mat).astype(np.float32)

        proprio_10 = np.concatenate([ee_pos, rot6d, np.array([0.0], dtype=np.float32)])
        proprio = np.concatenate([proprio_10, np.zeros(10, dtype=np.float32)])

        # --- processor: images + text -> input_ids, image_input, image_mask ---
        inputs = self.processor(images, self.instruction)

        device = self.device
        dtype = self._dtype

        def to_model(t):
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        inputs = {k: to_model(v) for k, v in inputs.items()}
        inputs["proprio"] = to_model(
            torch.as_tensor(proprio).unsqueeze(0)
        )
        inputs["domain_id"] = torch.tensor(
            [self.DOMAIN_ID], dtype=torch.long, device=device,
        )

        # --- generate ---
        with torch.no_grad():
            raw = self.model.generate_actions(
                **inputs, steps=self.DENOISE_STEPS,
            )  # (1, num_actions, dim_action)

        actions_np = raw.squeeze(0).float().cpu().numpy()  # (30, 20)

        # --- post-process: 20D -> 7D absolute [pos3, axisangle3, grip1] ---
        pos = actions_np[:, :3]
        aa = _rotate6d_to_axisangle(actions_np[:, 3:9])
        grip = np.where(actions_np[:, 9] > 0.5, 1.0, -1.0)[:, np.newaxis]

        return np.concatenate([pos, aa, grip], axis=1).astype(np.float64)

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "X-VLA requires predict_from_obs() with the full LIBERO obs dict "
            "(needs robot0_eef_pos/quat for proprioception). Set "
            "use_obs_predict=True on the policy function."
        )

    def reset(self) -> None:
        self.instruction = None
