#!/usr/bin/env python3
"""
Test that all VLAs covered by the eval framework can work.

- External models: get_commands() returns non-empty, well-formed commands.
- Native models: load_model() and make_policy_fn() work (skip if openpi/LIBERO missing).
- Integration: run_libero_eval --model X --print_cmd exits 0 for every model.

Run from agent_attack_framework:
    python -m eval.test_all_vlas
    python -m pytest eval/test_all_vlas.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys

# Project root on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)


# All 6 paper models
NATIVE_IDS = ("openpi_pi05",)
EXTERNAL_IDS = (
    "openvla",
    "molmoact",
    "deepthinkvla",
    "ecot",
    "internvla_m1",
)
ALL_MODEL_IDS = NATIVE_IDS + EXTERNAL_IDS


def test_external_config_exists():
    """Every external model has a config."""
    from eval.external.configs import EXTERNAL_MODELS, get_external_config
    for mid in EXTERNAL_IDS:
        assert get_external_config(mid) is not None, f"Missing config for {mid}"
        assert mid in EXTERNAL_MODELS, f"Missing EXTERNAL_MODELS entry for {mid}"


def test_external_get_commands_returns_non_empty():
    """get_commands() returns a non-empty list for each external model."""
    from eval.external.run import get_commands
    # Use a fake repo path so template-based models have something to substitute
    repo = "/fake/repo/path"
    for mid in EXTERNAL_IDS:
        commands = get_commands(mid, repo_path=repo, episodes_per_task=2, seed=7)
        assert isinstance(commands, list), f"{mid}: get_commands should return list"
        assert len(commands) >= 1, f"{mid}: need at least one command, got {len(commands)}"
        for cmd in commands:
            assert isinstance(cmd, str) and len(cmd) > 10, f"{mid}: invalid command {cmd!r}"


def test_external_commands_contain_expected():
    """Generated commands mention the suite or libero where relevant."""
    from eval.external.run import get_commands
    repo = "/fake/repo"
    # OpenVLA: 4 commands, one per suite
    openvla_cmds = get_commands("openvla", repo_path=repo)
    assert len(openvla_cmds) == 4
    assert all("libero_" in c for c in openvla_cmds)


def test_native_registry():
    """Native model ids are in NATIVE_MODELS and model_registry accepts them."""
    from eval.model_registry import NATIVE_MODELS
    for mid in NATIVE_IDS:
        assert mid in NATIVE_MODELS, f"{mid} should be in NATIVE_MODELS"


def test_native_load_model_imports():
    """Native load_model and make_policy_fn can be called (skip if openpi missing)."""
    from eval.model_registry import load_model, make_policy_fn
    try:
        model = load_model("openpi_pi05", replan_steps=2)
        assert model is not None
        assert hasattr(model, "set_language") and hasattr(model, "predict")
        policy_fn = make_policy_fn(model, "pick the bowl", replan_steps=2)
        assert callable(policy_fn)
    except Exception as e:
        err = str(e).lower()
        if "openpi" in err or "cannot import" in err or "no module" in err or "not found" in err:
            if _has_pytest:
                pytest.skip(f"OpenPI not available: {e}")  # noqa: F821
            raise _SkipTest(f"OpenPI not available: {e}")
        raise


def test_run_libero_eval_print_cmd_exits_zero():
    """run_libero_eval --model X --print_cmd exits 0 for every model."""
    for mid in ALL_MODEL_IDS:
        result = subprocess.run(
            [sys.executable, "-m", "eval.run_libero_eval", "--model", mid, "--print_cmd"],
            cwd=_FRAMEWORK_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"model={mid} exit={result.returncode} stderr={result.stderr!r}"
        )
        # Native: expect "Native model"; external: expect command-like output
        if mid in NATIVE_IDS:
            assert "Native model" in result.stdout or "openpi" in result.stdout.lower(), (
                f"model={mid}: expected native confirmation, got: {result.stdout[:200]!r}"
            )
        else:
            assert "REPO_PATH" in result.stdout or "lerobot-eval" in result.stdout or "cd " in result.stdout or "python " in result.stdout, (
                f"model={mid}: expected command output, got: {result.stdout[:200]!r}"
            )


class _SkipTest(Exception):
    pass


try:
    import pytest
    _has_pytest = True
except ImportError:
    _has_pytest = False


def run_standalone():
    """Run tests without pytest."""
    failed = []
    tests = [
        ("external_config_exists", test_external_config_exists),
        ("external_get_commands_non_empty", test_external_get_commands_returns_non_empty),
        ("external_commands_expected", test_external_commands_contain_expected),
        ("native_registry", test_native_registry),
        ("native_load_model", test_native_load_model_imports),
        ("run_libero_eval_print_cmd", test_run_libero_eval_print_cmd_exits_zero),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except _SkipTest as e:
            print(f"  SKIP {name}: {e}")
        except BaseException as e:
            if type(e).__name__ == "Skipped":
                print(f"  SKIP {name}: {e}")
            elif isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            else:
                print(f"  FAIL {name}: {e}")
                failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_standalone())
