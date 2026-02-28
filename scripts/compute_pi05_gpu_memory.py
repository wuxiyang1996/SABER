#!/usr/bin/env python3
"""
Compute GPU memory used by Pi0.5-LIBERO (OpenPI flow-matching VLA).

Run from agent_attack_framework/ with a single GPU visible, e.g.:
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_pi05_gpu_memory.py

Or from repo root:
  cd agent_attack_framework && CUDA_VISIBLE_DEVICES=0 python scripts/compute_pi05_gpu_memory.py

Prints:
  - GPU memory before loading the model
  - GPU memory after loading Pi0.5 and running one inference
  - Delta (memory attributable to Pi0.5 inference)

Documented estimate (from train_vla.py): Pi0.5 inference needs ~9 GiB.
"""

from __future__ import annotations

import os
import subprocess
import sys


def get_gpu_memory_mb(device_index: int = 0) -> int | None:
    """Return used GPU memory in MiB for the given device, or None if nvidia-smi fails."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().split("\n")[0].strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: could not query GPU memory: {e}", file=sys.stderr)
        return None


def main() -> int:
    project_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    libero_rollouts = os.path.join(project_root, "libero_rollouts")
    if libero_rollouts not in sys.path:
        sys.path.insert(0, libero_rollouts)

    # Match framework: limit JAX to a fraction so we measure actual usage, not prealloc
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    mem_before = get_gpu_memory_mb(0)
    if mem_before is not None:
        print(f"GPU memory before loading Pi0.5: {mem_before} MiB ({mem_before / 1024:.2f} GiB)")

    try:
        from pi05_libero_model import Pi05LiberoModel, preprocess_image, build_libero_state
        import numpy as np
    except ModuleNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Ensure openpi and libero are installed and ROBOTWIN_ROOT or openpi/src is set.", file=sys.stderr)
        return 1

    print("Loading Pi0.5-LIBERO model...")
    model = Pi05LiberoModel(train_config_name="pi05_libero")
    model.set_language("pick up the block")

    # Dummy observation: 224x224 images + 8-dim state (as in LIBERO)
    h, w = 224, 224
    dummy_obs = {
        "agentview_image": np.zeros((h, w, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((h, w, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
    }
    print("Running one inference...")
    _ = model.predict_from_obs(dummy_obs)

    mem_after = get_gpu_memory_mb(0)
    if mem_after is not None:
        print(f"GPU memory after load + 1 inference: {mem_after} MiB ({mem_after / 1024:.2f} GiB)")
        if mem_before is not None:
            delta = mem_after - mem_before
            print(f"Delta (Pi0.5 load + inference): {delta} MiB ({delta / 1024:.2f} GiB)")

    print("\nDocumented estimate (train_vla.py): Pi0.5 inference ~9 GiB.")
    print("Use XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 (VLA GPUs are dedicated, not shared with vLLM).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
