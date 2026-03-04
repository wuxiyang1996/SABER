#!/usr/bin/env python3
"""VLA model subprocess server.

Runs in an isolated conda environment (e.g. vla_models) with the correct
transformers / tokenizers versions for each VLA model.  Communicates with the
parent process via length-prefixed pickle over stdin/stdout.

Protocol (binary, over stdin/stdout):
    Request:  [4-byte big-endian length][pickled dict]
    Response: [4-byte big-endian length][pickled dict]

Commands:
    {"cmd": "init", "model_id": str, "suite_name": str, "checkpoint": str|None,
     "device": str, "action_horizon": int, "replan_steps": int}
    {"cmd": "set_language", "instruction": str}
    {"cmd": "predict", "agentview": np.ndarray, "wrist": np.ndarray, "state": np.ndarray}
    {"cmd": "predict_from_obs", "obs": dict}
    {"cmd": "reset"}
    {"cmd": "shutdown"}

Usage:
    /path/to/vla_models/bin/python vla_subprocess_server.py
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
import traceback

# Redirect model-library chatter to stderr so it doesn't corrupt the protocol
_real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
_real_stdin = os.fdopen(os.dup(sys.stdin.fileno()), "rb")
sys.stdout = open(os.devnull, "w")
sys.stdin = open(os.devnull, "r")


def _recv(stream) -> dict:
    raw_len = stream.read(4)
    if len(raw_len) < 4:
        raise EOFError("stdin closed")
    length = struct.unpack(">I", raw_len)[0]
    data = bytearray()
    while len(data) < length:
        chunk = stream.read(length - len(data))
        if not chunk:
            raise EOFError("stdin closed mid-message")
        data.extend(chunk)
    return pickle.loads(bytes(data))


def _send(stream, msg: dict):
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    # Tell model_factory we are already inside a subprocess — load in-process,
    # do NOT spawn another SubprocessVLAWrapper (would recurse infinitely).
    os.environ["_VLA_SUBPROCESS_SERVER"] = "1"

    model = None
    _send(_real_stdout, {"status": "ready"})

    while True:
        try:
            req = _recv(_real_stdin)
        except EOFError:
            break

        cmd = req.get("cmd")
        try:
            if cmd == "init":
                from model_factory import load_vla_model
                model = load_vla_model(
                    model_id=req["model_id"],
                    suite_name=req.get("suite_name"),
                    checkpoint_path=req.get("checkpoint"),
                    device=req.get("device", "cuda:0"),
                    action_horizon=req.get("action_horizon"),
                    replan_steps=req.get("replan_steps", 5),
                )
                _send(_real_stdout, {"status": "ok"})

            elif cmd == "set_language":
                model.set_language(req["instruction"])
                _send(_real_stdout, {"status": "ok"})

            elif cmd == "predict":
                import numpy as _np
                _agentview = _np.asarray(req["agentview"], dtype=_np.uint8)
                _wrist = _np.asarray(req["wrist"], dtype=_np.uint8)
                _state = _np.asarray(req["state"], dtype=_np.float64)
                action = model.predict(_agentview, _wrist, _state)
                _send(_real_stdout, {
                    "status": "ok",
                    "action": action.tolist(),
                    "action_shape": list(action.shape),
                })

            elif cmd == "predict_from_obs":
                action = model.predict_from_obs(req["obs"])
                _send(_real_stdout, {
                    "status": "ok",
                    "action": action.tolist(),
                    "action_shape": list(action.shape),
                })

            elif cmd == "reset":
                if model is not None:
                    model.reset()
                _send(_real_stdout, {"status": "ok"})

            elif cmd == "shutdown":
                _send(_real_stdout, {"status": "ok"})
                break

            else:
                _send(_real_stdout, {"status": "error", "error": f"unknown cmd: {cmd}"})

        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            _send(_real_stdout, {"status": "error", "error": tb})

    _real_stdout.close()
    _real_stdin.close()


if __name__ == "__main__":
    main()
