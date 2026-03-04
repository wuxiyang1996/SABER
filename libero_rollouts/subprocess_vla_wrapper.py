"""Subprocess VLA wrapper.

Proxies VLA model calls to a child process running in an isolated conda
environment (e.g. ``vla_models``) that has the correct ``transformers`` /
``tokenizers`` versions.  The attack agent stays in the main environment
(``runpod``).

Usage::

    wrapper = SubprocessVLAWrapper(
        python="/workspace/miniforge3/envs/vla_models/bin/python",
        model_id="openvla",
        suite_name="libero_spatial",
        device="cuda:0",
    )
    wrapper.set_language("pick up the red block")
    action = wrapper.predict(agentview, wrist, state)
    wrapper.shutdown()
"""

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

_SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vla_subprocess_server.py",
)


class SubprocessVLAWrapper:
    def __init__(
        self,
        python: str,
        model_id: str,
        suite_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "cuda:0",
        action_horizon: Optional[int] = None,
        replan_steps: int = 5,
    ):
        self.model_id = model_id
        self.suite_name = suite_name
        self.python = python

        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env.setdefault("TORCH_HOME", "/workspace/.cache_torch")
        env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
        env.setdefault("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
        env.setdefault("HF_LEROBOT_HOME", "/workspace/.cache/lerobot")

        # Prevent the subprocess server from spawning yet another subprocess.
        env["_VLA_SUBPROCESS_SERVER"] = "1"

        # Remove parent conda env paths that can contaminate the child env's
        # C-extension loading (e.g. numpy PyCapsule_Import failures).
        vla_env_root = os.path.dirname(os.path.dirname(python))  # .../envs/vla_models
        for path_var in ("LD_LIBRARY_PATH", "PYTHONPATH"):
            old = env.get(path_var, "")
            if old:
                cleaned = os.pathsep.join(
                    p for p in old.split(os.pathsep)
                    if not p or "/envs/" not in p or p.startswith(vla_env_root)
                )
                env[path_var] = cleaned

        # Add Isaac-GR00T repo to PYTHONPATH for groot model loading
        groot_repo = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "repos", "groot",
        )
        if os.path.isdir(groot_repo):
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{groot_repo}:{pp}" if pp else groot_repo

        self._proc = subprocess.Popen(
            [python, _SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
        )
        ready = self._recv()
        assert ready.get("status") == "ready", f"Server failed to start: {ready}"

        resp = self._call({
            "cmd": "init",
            "model_id": model_id,
            "suite_name": suite_name,
            "checkpoint": checkpoint,
            "device": device,
            "action_horizon": action_horizon,
            "replan_steps": replan_steps,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"VLA subprocess init failed: {resp.get('error', resp)}")

    def _send(self, msg: dict):
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack(">I", len(payload)))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        raw_len = self._proc.stdout.read(4)
        if len(raw_len) < 4:
            raise EOFError("VLA subprocess closed unexpectedly")
        length = struct.unpack(">I", raw_len)[0]
        data = bytearray()
        while len(data) < length:
            chunk = self._proc.stdout.read(length - len(data))
            if not chunk:
                raise EOFError("VLA subprocess closed mid-message")
            data.extend(chunk)
        return pickle.loads(bytes(data))

    def _call(self, msg: dict) -> dict:
        self._send(msg)
        resp = self._recv()
        if resp.get("status") == "error":
            raise RuntimeError(f"VLA subprocess error: {resp['error']}")
        return resp

    def set_language(self, instruction: str) -> None:
        self._call({"cmd": "set_language", "instruction": instruction})

    def predict(
        self,
        agentview_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        resp = self._call({
            "cmd": "predict",
            "agentview": agentview_image.tolist(),
            "wrist": wrist_image.tolist(),
            "state": state.tolist(),
        })
        return np.array(resp["action"], dtype=np.float64).reshape(resp["action_shape"])

    def predict_from_obs(self, obs: dict) -> np.ndarray:
        resp = self._call({"cmd": "predict_from_obs", "obs": obs})
        return np.array(resp["action"], dtype=np.float64).reshape(resp["action_shape"])

    def reset(self) -> None:
        self._call({"cmd": "reset"})

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"cmd": "shutdown"})
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
                self._proc.wait()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
