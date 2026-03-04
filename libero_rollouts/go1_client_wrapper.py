"""GO-1 (AgiBot-World) client wrapper for LIBERO replay evaluation.

Uses the HTTP server-client setup from OpenDriveLab/AgiBot-World:
  - Server: run deploy.py (GO-1 model) with --model_path and --data_stats_path.
  - This wrapper is the client: it sends observations to POST /act and gets actions.

Payload format (must match AgiBot-World evaluate/libero/main.py):
  - "top": agentview image (H,W,3) uint8, rotated 180
  - "left": wrist image (H,W,3) uint8, rotated 180
  - "instruction": str
  - "state": (1, 7) or (1, 8) — eef_pos(3) + axis_angle(3) + gripper(1 or 2)
  - "ctrl_freqs": [10]

Server returns a list of actions (action chunk); we return as (N, 7) numpy array.

See: https://github.com/OpenDriveLab/AgiBot-World/blob/main/evaluate/libero/README.md
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from pi05_libero_model import quat2axisangle

# Default server URL (host:port). Override with GO1_SERVER env or checkpoint param.
_DEFAULT_SERVER = "127.0.0.1:9000"


def _build_go1_state(obs: dict) -> np.ndarray:
    """Build state vector for GO-1 server: eef_pos(3) + axis_angle(3) + gripper (1 or 2)."""
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    axis_angle = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).flatten()
    return np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float64).reshape(1, -1)


class GO1ClientWrapper:
    def __init__(
        self,
        checkpoint: str = _DEFAULT_SERVER,
        suite_name: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: int = 5,
        replan_steps: int = 5,
    ):
        # checkpoint is used as server URL (host:port)
        self.server_url = os.environ.get("GO1_SERVER", checkpoint).strip()
        if not self.server_url.startswith("http"):
            self.server_url = f"http://{self.server_url}"
        self._act_url = f"{self.server_url.rstrip('/')}/act"
        self.suite_name = suite_name
        self.action_horizon = action_horizon
        self.replan_steps = replan_steps
        self.instruction: Optional[str] = None
        self.use_obs_predict = True  # Eval uses predict_from_obs so we send 256x256 to server
        print(f"[GO1] Client wrapper for {self._act_url} (action_horizon={action_horizon})")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        import requests

        if self.instruction is None:
            raise RuntimeError("Call set_language(instruction) before predict()")

        # Match AgiBot main.py: images rotated 180, uint8
        top = np.ascontiguousarray(agentview_image[::-1, ::-1])
        if top.dtype != np.uint8 and np.issubdtype(top.dtype, np.floating):
            top = (255 * top).astype(np.uint8)
        left = np.ascontiguousarray(wrist_image[::-1, ::-1])
        if left.dtype != np.uint8 and np.issubdtype(left.dtype, np.floating):
            left = (255 * left).astype(np.uint8)

        state_1n = np.asarray(state, dtype=np.float64)
        if state_1n.ndim == 1:
            state_1n = state_1n.reshape(1, -1)

        payload = {
            "top": top.tolist(),
            "left": left.tolist(),
            "instruction": self.instruction,
            "state": state_1n.tolist(),
            "ctrl_freqs": [10],
        }

        try:
            resp = requests.post(
                self._act_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"GO-1 server not reachable at {self._act_url}. "
                "Start the server first (e.g. conda activate go1 && python evaluate/deploy.py "
                "--model_path ... --data_stats_path ... --port 9000), or run the eval script with "
                "GO1_MODEL_PATH and GO1_DATA_STATS_PATH set to start the server automatically. "
                f"Original error: {e}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(
                f"GO-1 server request failed ({self._act_url}): {e}"
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"GO-1 server {self._act_url} returned {resp.status_code}: {resp.text[:500]}"
            )
        action = np.array(resp.json(), dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        return action

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        top = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        if top.dtype != np.uint8 and np.issubdtype(top.dtype, np.floating):
            top = (255 * top).astype(np.uint8)
        left = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        if left.dtype != np.uint8 and np.issubdtype(left.dtype, np.floating):
            left = (255 * left).astype(np.uint8)
        state = _build_go1_state(obs)
        return self.predict(top, left, state)

    def reset(self) -> None:
        self.instruction = None
