#!/usr/bin/env python3
"""
Check that the libero conda env has pi0.5 and agentic attack dependencies for LIBERO.

Run from agent_attack_framework/ with:  python installation/check_libero_env.py
Or with conda:  conda run -n libero python agent_attack_framework/installation/check_libero_env.py
"""
from __future__ import annotations

import os
import sys

def main() -> int:
    project_root = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    inrepo_openpi_src = os.path.join(project_root, "openpi", "src")
    robotwin_root = os.environ.get(
        "ROBOTWIN_ROOT",
        os.path.realpath(os.path.join(project_root, "..", "RoboTwin")),
    )
    pi05_policy = os.path.join(robotwin_root, "policy", "pi05")
    pi05_src = os.path.join(pi05_policy, "src")

    errors: list[str] = []
    warnings: list[str] = []

    # Python version
    ver = sys.version_info
    if (ver.major, ver.minor) != (3, 11):
        warnings.append(f"Python {ver.major}.{ver.minor} (recommended: 3.11)")
    else:
        print(f"[OK] Python {ver.major}.{ver.minor}")

    # Pi0.5 / openpi: in-repo (agent_attack_framework/openpi/src) or RoboTwin
    use_inrepo = (
        os.environ.get("ROBOTWIN_ROOT") is None
        and os.path.isdir(inrepo_openpi_src)
        and os.path.isdir(os.path.join(inrepo_openpi_src, "openpi"))
    )
    if use_inrepo:
        print(f"[OK] openpi (in-repo): {inrepo_openpi_src}")
        if inrepo_openpi_src not in sys.path:
            sys.path.insert(0, inrepo_openpi_src)
        inrepo_openpi_client_src = os.path.join(project_root, "openpi", "packages", "openpi-client", "src")
        if os.path.isdir(inrepo_openpi_client_src) and inrepo_openpi_client_src not in sys.path:
            sys.path.insert(0, inrepo_openpi_client_src)
    else:
        if not os.path.isdir(robotwin_root):
            errors.append(f"RoboTwin root not found: {robotwin_root} (or use in-repo openpi at agent_attack_framework/openpi/src)")
        else:
            print(f"[OK] RoboTwin root: {robotwin_root}")
        if not os.path.isdir(pi05_policy):
            errors.append(f"Pi0.5 policy dir not found: {pi05_policy}")
        else:
            print(f"[OK] Pi0.5 policy dir: {pi05_policy}")
        if not os.path.isdir(pi05_src):
            errors.append(f"Pi0.5 src dir not found: {pi05_src}")
        else:
            print(f"[OK] Pi0.5 src dir: {pi05_src}")
            if pi05_src not in sys.path:
                sys.path.insert(0, pi05_src)
        if pi05_policy not in sys.path:
            sys.path.insert(0, pi05_policy)

    # JAX / OpenPI (for Pi0.5)
    try:
        import jax
        print(f"[OK] jax {jax.__version__}")
    except ImportError as e:
        errors.append(f"jax: {e}")
        jax = None  # type: ignore[assignment]
    try:
        import flax
        print(f"[OK] flax {flax.__version__}")
    except ImportError as e:
        errors.append(f"flax: {e}")
    try:
        import openpi
        print("[OK] openpi")
    except ImportError as e:
        errors.append(f"openpi: {e}")

    # LIBERO
    try:
        import libero.libero
        print("[OK] libero")
    except ImportError as e:
        errors.append(f"libero: {e}")

    # Agentic attack (openpipe-art -> 'art', vLLM, LangGraph)
    try:
        import art
        print("[OK] art (openpipe-art agentic attack backend)")
    except ImportError as e:
        errors.append(f"art (openpipe-art): {e}")
    try:
        import vllm
        print(f"[OK] vllm {vllm.__version__}")
        if getattr(vllm, "__version__", "") != "0.13.0":
            warnings.append(f"vllm {vllm.__version__} (requirements.txt pins 0.13.0)")
    except ImportError as e:
        errors.append(f"vllm: {e}")
    try:
        import langgraph
        print("[OK] langgraph")
    except ImportError as e:
        errors.append(f"langgraph: {e}")

    # GPU availability (required for π0.5 via JAX and vLLM)
    try:
        import torch

        print(f"[OK] torch {torch.__version__}")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            errors.append(
                "CUDA/GPU not available to PyTorch "
                f"(cuda.is_available()={torch.cuda.is_available()}, "
                f"device_count={torch.cuda.device_count()})."
            )
    except ImportError as e:
        errors.append(f"torch: {e}")

    if jax is not None:
        try:
            gpus = jax.devices("gpu")
            if len(gpus) < 1:
                errors.append("JAX sees 0 GPU devices (only CPU backend available).")
            else:
                print(f"[OK] jax GPU devices: {gpus}")
        except Exception as e:
            errors.append(f"JAX GPU backend not available: {e}")

    if not any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidiactl")):
        warnings.append(
            "No /dev/nvidia* device nodes detected. If you expect GPUs, ensure your "
            "job/container exposes NVIDIA devices."
        )

    # Optional: Pi0.5 wrapper import (adds RoboTwin/policy/pi05 to path)
    sys.path.insert(0, project_root)
    libero_rollouts = os.path.join(project_root, "libero_rollouts")
    if libero_rollouts not in sys.path:
        sys.path.insert(0, libero_rollouts)
    try:
        from pi05_libero_model import Pi05LiberoModel
        print("[OK] Pi05LiberoModel (pi0.5 LIBERO wrapper)")
    except Exception as e:
        warnings.append(f"Pi05LiberoModel import: {e}")

    print()
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[FAIL] {e}")
    if errors:
        print("\nFix the [FAIL] items above. See requirements.txt and RUN.md.")
        return 1
    if warnings:
        print("Warnings above are optional to fix.")
    else:
        print("libero env is ready for pi0.5 + agentic attack in LIBERO.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
