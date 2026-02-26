#!/usr/bin/env python3
"""
Apply ART 0.5.x ↔ vLLM 0.11.x compatibility patches.

Three patches:
  1. Add pause_generation / resume_generation stubs to AsyncLLM
     (ART calls them; vLLM < 0.16 does not have them).
  2. Replace run_on_workers(do_sleep/do_wake_up) with native
     llm.sleep() / llm.wake_up() in ART's UnslothService.train()
     (avoids EngineDeadError from bypassing EngineCore coordination).
  3. Fix tool_parsers import path in ART's patches.py
     (vLLM 0.11 has it under vllm.entrypoints.openai.tool_parsers,
      not vllm.tool_parsers).

Usage:
    python scripts/apply_vllm_patches.py          # auto-detect site-packages
    python scripts/apply_vllm_patches.py --check   # dry-run: report status only
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys


def _site_packages() -> str:
    """Return the site-packages directory for the active Python."""
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            return p
    raise RuntimeError("Could not find site-packages in sys.path")


PAUSE_RESUME_STUB = '''\

    # -- ART compat (added by apply_vllm_patches.py, native in vLLM >=0.16) --
    async def pause_generation(self, mode: str = "keep") -> None:
        pass

    async def resume_generation(self) -> None:
        pass
'''


def patch_async_llm(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 1: add pause_generation / resume_generation stubs."""
    path = os.path.join(sp, "vllm", "v1", "engine", "async_llm.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (vLLM not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    if "pause_generation" in src:
        print("[OK]   Patch 1: pause_generation / resume_generation already present")
        return False

    anchor = "    async def sleep(self"
    if anchor not in src:
        print(f"[FAIL] Patch 1: could not find anchor '{anchor}' in {path}")
        return False

    if dry_run:
        print("[NEED] Patch 1: pause_generation / resume_generation stubs missing")
        return True

    patched = src.replace(anchor, PAUSE_RESUME_STUB + "\n" + anchor, 1)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 1: added pause_generation / resume_generation stubs to {path}")
    return True


def patch_unsloth_service(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 2: replace run_on_workers sleep/wake with native llm.sleep/wake_up."""
    path = os.path.join(sp, "art", "unsloth", "service.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (ART not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    old_sleep = "await run_on_workers(llm, do_sleep, level=sleep_level)"
    old_wake = "await run_on_workers(llm, do_wake_up)"

    needs_sleep = old_sleep in src
    needs_wake = old_wake in src

    if not needs_sleep and not needs_wake:
        print("[OK]   Patch 2: sleep/wake already uses native vLLM pipeline")
        return False

    if dry_run:
        print("[NEED] Patch 2: run_on_workers(do_sleep/do_wake_up) needs replacing")
        return True

    patched = src
    if needs_sleep:
        patched = patched.replace(old_sleep, "await llm.sleep(sleep_level)")
    if needs_wake:
        patched = patched.replace(old_wake, "await llm.wake_up()")

    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 2: replaced sleep/wake with native vLLM pipeline in {path}")
    return True


def patch_tool_parser_import(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 3: fix tool_parsers import path for vLLM 0.11.x."""
    path = os.path.join(sp, "art", "vllm", "patches.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (ART not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    old_import = "from vllm.tool_parsers.abstract_tool_parser import ToolParserManager"
    new_import = "from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager"

    if old_import not in src:
        if new_import in src:
            print("[OK]   Patch 3: tool_parsers import path already fixed")
        else:
            print("[OK]   Patch 3: tool_parsers import not present (nothing to fix)")
        return False

    if dry_run:
        print("[NEED] Patch 3: tool_parsers import path needs fixing (vllm.tool_parsers → vllm.entrypoints.openai.tool_parsers)")
        return True

    patched = src.replace(old_import, new_import)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 3: fixed tool_parsers import path in {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Dry-run: report status only")
    args = parser.parse_args()

    sp = _site_packages()
    print(f"site-packages: {sp}\n")

    p1 = patch_async_llm(sp, dry_run=args.check)
    p2 = patch_unsloth_service(sp, dry_run=args.check)
    p3 = patch_tool_parser_import(sp, dry_run=args.check)

    if args.check:
        if p1 or p2 or p3:
            print("\nRun without --check to apply patches.")
            sys.exit(1)
        else:
            print("\nAll patches already applied.")
    else:
        if not p1 and not p2 and not p3:
            print("\nNothing to patch.")


if __name__ == "__main__":
    main()
