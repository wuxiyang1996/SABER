#!/usr/bin/env python3
"""
Apply ART 0.5.x ↔ vLLM 0.11.x compatibility patches.

Six patches:
  1. Add pause_generation / resume_generation stubs to AsyncLLM
     (ART calls them; vLLM < 0.16 does not have them).
  2. Replace run_on_workers(do_sleep/do_wake_up) with native
     llm.sleep() / llm.wake_up() in ART's UnslothService.train()
     (avoids EngineDeadError from bypassing EngineCore coordination).
  2b. Before wake_up(): sync GPU, gc, empty_cache(5), sleep(2); catch
      EngineDeadError and re-raise as RuntimeError (reduces OOM during training).
  3. Fix tool_parsers import path in ART's patches.py
     (vLLM 0.11 has it under vllm.entrypoints.openai.tool_parsers,
      not vllm.tool_parsers).
  4. In UnslothService._state: if config has "training_device", move
     model and peft_model to that device so inference (vLLM) and
     training (Unsloth) can use different GPUs (e.g. --attack_gpus 2,3).
  5. In UnslothService.train(): when training_device is set (2-GPU split),
     skip vLLM sleep/wake entirely. Training runs on a separate GPU so
     vLLM's memory is never touched — eliminates OOM from fragmentation.

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


# Block inserted before llm.wake_up() to reduce OOM: release GPU memory and catch EngineDeadError.
# Uses 4-space base indent so we can re-indent to match ART's service.py.
_WAKE_UP_OOM_FIX_BLOCK = """    import torch
    import gc
    torch.cuda.synchronize()
    gc.collect()
    for _ in range(5):
        torch.cuda.empty_cache()
    await asyncio.sleep(2.0)
    try:
        await llm.wake_up()
    except Exception as wake_err:
        from vllm.v1.engine.exceptions import EngineDeadError
        if isinstance(wake_err, EngineDeadError):
            raise RuntimeError(
                "vLLM EngineCore died during training (often OOM or worker crash). "
                "Restart with: python train_vla.py --resume . "
                "Lower --gpu_memory_utilization (e.g. 0.60) to leave more headroom, "
                "or check dmesg for OOM killer."
            ) from wake_err
        raise
"""


def _reindent_block(block: str, indent: str) -> str:
    """Reindent block (4-space base) to use given indent; preserve nested 4-space level."""
    lines = block.strip().split("\n")
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if line.startswith("        "):  # 8 spaces -> nested
            out.append(indent + "    " + line[8:])
        elif line.startswith("    "):  # 4 spaces -> top level
            out.append(indent + line[4:])
        else:
            out.append(indent + line)
    return "\n".join(out)


def patch_unsloth_service(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 2: replace run_on_workers sleep/wake with native llm.sleep/wake_up.
    Patch 2b: before wake_up(), sync + gc + empty_cache + sleep; catch EngineDeadError."""
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

    # Already patched with OOM fix (our block contains this message)
    has_oom_fix = "vLLM EngineCore died during training" in src

    if not needs_sleep and not needs_wake and has_oom_fix:
        print("[OK]   Patch 2: sleep/wake + OOM fix already applied")
        return False

    if not needs_sleep and not needs_wake and not has_oom_fix:
        # Native sleep/wake already there; add OOM fix only (replace bare "await llm.wake_up()")
        match = re.search(r"^(\s+)await llm\.wake_up\(\)\s*$", src, re.MULTILINE)
        if not match:
            print("[OK]   Patch 2: sleep/wake already native (wake_up pattern not found for OOM fix)")
            return False
        if dry_run:
            print("[NEED] Patch 2b: add sync/gc/sleep before wake_up and catch EngineDeadError")
            return True
        indent = match.group(1)
        block = _reindent_block(_WAKE_UP_OOM_FIX_BLOCK, indent)
        patched = src.replace(match.group(0), block, 1)
        with open(path, "w") as f:
            f.write(patched)
        print(f"[DONE] Patch 2b: added OOM fix (sync/gc/sleep + EngineDeadError handling) in {path}")
        return True

    if dry_run:
        print("[NEED] Patch 2: run_on_workers(do_sleep/do_wake_up) needs replacing (+ OOM fix)")
        return True

    patched = src
    if needs_sleep:
        patched = patched.replace(old_sleep, "await llm.sleep(sleep_level)")
    if needs_wake:
        # Replace with OOM-fix block (same indentation as original await llm.wake_up())
        match = re.search(r"^(\s+)await run_on_workers\(llm, do_wake_up\)\s*$", patched, re.MULTILINE)
        if match:
            indent = match.group(1)
            block = _reindent_block(_WAKE_UP_OOM_FIX_BLOCK, indent)
            patched = patched.replace(match.group(0), block, 1)
        else:
            patched = patched.replace(old_wake, "await llm.wake_up()")

    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 2: replaced sleep/wake with native vLLM pipeline + OOM fix in {path}")
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


# Block inserted before "return UnslothState(" so Unsloth training can use a different GPU.
_TRAINING_DEVICE_BLOCK = """        training_device = self.config.get("training_device")
        if training_device:
            model = model.to(training_device)
            peft_model = peft_model.to(training_device)

"""


def patch_unsloth_training_device(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 4: in UnslothService._state, move model/peft_model to config['training_device'] when set."""
    path = os.path.join(sp, "art", "unsloth", "service.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (ART not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    anchor = "        trainer._prepare_inputs = _async_prepare_inputs\n\n        return UnslothState("
    if _TRAINING_DEVICE_BLOCK.strip() in src and "training_device = self.config.get" in src:
        print("[OK]   Patch 4: training_device placement already present")
        return False

    if anchor not in src:
        print("[OK]   Patch 4: anchor not found (ART structure changed?)")
        return False

    if dry_run:
        print("[NEED] Patch 4: add training_device placement in UnslothService._state")
        return True

    patched = src.replace(
        anchor,
        "        trainer._prepare_inputs = _async_prepare_inputs\n\n"
        + _TRAINING_DEVICE_BLOCK
        + "\n        return UnslothState(",
        1,
    )
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 4: added training_device placement in {path}")
    return True


def patch_split_gpu_sleep_wake(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 5: skip vLLM sleep/wake when training_device is set (2-GPU split).

    When training and inference run on separate GPUs, vLLM doesn't need to
    sleep/wake because training never touches the inference GPU's memory.
    This eliminates the OOM from the sleep/wake fragmentation cycle.
    """
    path = os.path.join(sp, "art", "unsloth", "service.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (ART not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    if '_split_gpu = training_device is not None' in src:
        print("[OK]   Patch 5: split-GPU sleep/wake bypass already present")
        return False

    # Look for the train method's llm assignment
    anchor = "        llm = await self.llm\n\n        # Pause generation"
    if anchor not in src:
        anchor = "        llm = await self.llm\n        training_device = self.config.get"
        if anchor in src:
            print("[OK]   Patch 5: split-GPU sleep/wake bypass already present (alt check)")
            return False
        print("[SKIP] Patch 5: could not find anchor in train() method")
        return False

    if dry_run:
        print("[NEED] Patch 5: add split-GPU sleep/wake bypass in train()")
        return True

    # Replace the train method's preamble and sleep/wake sections
    patched = src

    # 1. Add training_device lookup after llm assignment
    patched = patched.replace(
        "        llm = await self.llm\n\n        # Pause generation to prevent new requests during training\n        await llm.pause_generation()",
        "        llm = await self.llm\n"
        "        training_device = self.config.get(\"training_device\")\n"
        "        _split_gpu = training_device is not None\n\n"
        "        # Pause generation to prevent new requests during training\n"
        "        await llm.pause_generation()",
        1,
    )

    # 2. Guard the sleep section
    old_sleep_block = (
        "        # Always use level 1 sleep so that model weights are offloaded to CPU\n"
        "        # and properly restored on wake_up.  Level 2 discards weights without\n"
        "        # backup, leaving vLLM with uninitialized GPU memory after wake_up.\n"
        "        if not llm.output_processor.has_unfinished_requests():\n"
        "            await llm.reset_prefix_cache()\n"
        "        sleep_level = 1\n\n"
        "        # Put workers to sleep\n"
        "        await llm.sleep(sleep_level)\n"
        "        self._is_sleeping = True\n"
        "        gc_and_empty_cuda_cache()\n\n"
        "        # Reload training model to GPU (after vLLM is asleep)\n"
        "        self._state.reload_to_gpu()"
    )
    new_sleep_block = (
        "        if not _split_gpu:\n"
        "            # Single-GPU: sleep vLLM to free GPU memory for training\n"
        "            if not llm.output_processor.has_unfinished_requests():\n"
        "                await llm.reset_prefix_cache()\n"
        "            sleep_level = 1\n"
        "            await llm.sleep(sleep_level)\n"
        "            self._is_sleeping = True\n"
        "            gc_and_empty_cuda_cache()\n\n"
        "        # Reload training model to GPU (after vLLM is asleep)\n"
        "        self._state.reload_to_gpu(device=training_device or \"cuda:0\")"
    )
    if old_sleep_block in patched:
        patched = patched.replace(old_sleep_block, new_sleep_block, 1)
    else:
        print("[WARN] Patch 5: could not find sleep block to guard")

    # 3. Guard the wake section — look for the offload + wake block
    old_wake_block = (
        "        # Offload training model to CPU before waking vLLM\n"
        "        self._state.offload_to_cpu()\n\n"
        "        # Aggressive cleanup so vLLM workers can restore weights to GPU without OOM.\n"
        "        # Sync CUDA, then free memory and wait for any pending ops to complete.\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.synchronize()\n"
        "        gc_and_empty_cuda_cache(5)\n"
        "        await asyncio.sleep(4.0)\n"
    )
    new_wake_block = (
        "        # Offload training model to CPU before waking vLLM\n"
        "        self._state.offload_to_cpu()\n\n"
        "        if not _split_gpu:\n"
        "            # Single-GPU: aggressive cleanup so vLLM can restore weights\n"
        "            if torch.cuda.is_available():\n"
        "                torch.cuda.synchronize()\n"
        "            gc_and_empty_cuda_cache(5)\n"
        "            await asyncio.sleep(4.0)\n"
    )
    if old_wake_block in patched:
        patched = patched.replace(old_wake_block, new_wake_block, 1)
    else:
        print("[WARN] Patch 5: could not find wake block to guard")

    # 4. Guard the wake_up call and EngineDeadError handling
    old_wake_call = (
        "        # Wake up workers (may raise EngineDeadError if EngineCore died during training)\n"
        "        try:\n"
        "            await llm.wake_up()\n"
        "        except Exception as e:\n"
        "            if _EngineDeadError is not None and isinstance(e, _EngineDeadError):\n"
        "                raise RuntimeError(\n"
    )
    new_wake_call = (
        "            # Wake up workers (may raise EngineDeadError if EngineCore died)\n"
        "            try:\n"
        "                await llm.wake_up()\n"
        "            except Exception as e:\n"
        "                if _EngineDeadError is not None and isinstance(e, _EngineDeadError):\n"
        "                    raise RuntimeError(\n"
    )
    if old_wake_call in patched:
        patched = patched.replace(old_wake_call, new_wake_call, 1)

        # Also indent the rest of the exception block
        patched = patched.replace(
            '                    "vLLM EngineCore died during training (often OOM or worker crash). "\n'
            '                    "Restart with: python train_vla.py --resume . "\n'
            '                    "Lower --gpu_memory_utilization (e.g. 0.60) to leave more headroom, "\n'
            '                    "or check dmesg for OOM killer."\n'
            '                ) from e\n'
            '            raise\n'
            '        self._is_sleeping = False',
            '                        "vLLM EngineCore died during training (often OOM or worker crash). "\n'
            '                        "Restart with: python train_vla.py --resume . "\n'
            '                        "Lower --gpu_memory_utilization (e.g. 0.60) to leave more headroom, "\n'
            '                        "or check dmesg for OOM killer."\n'
            '                    ) from e\n'
            '                raise\n'
            '            self._is_sleeping = False',
            1,
        )
    else:
        print("[WARN] Patch 5: could not find wake_up call to guard")

    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 5: added split-GPU sleep/wake bypass in {path}")
    return True


def patch_get_model_config_training_device(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 6: preserve training_device through get_model_config().

    ART's get_model_config() rebuilds InternalModelConfig from scratch,
    dropping unknown keys like training_device.  This patch forwards it.
    """
    path = os.path.join(sp, "art", "dev", "get_model_config.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (ART not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    if 'config.get("training_device")' in src:
        print("[OK]   Patch 6: training_device forwarding already present")
        return False

    anchor = '    return InternalModelConfig(\n'
    if anchor not in src:
        print("[SKIP] Patch 6: anchor not found in get_model_config.py")
        return False

    if dry_run:
        print("[NEED] Patch 6: forward training_device through get_model_config()")
        return True

    # Replace "return InternalModelConfig(..." with "result = ...; forward; return result"
    old_return = (
        '    return InternalModelConfig(\n'
        '        init_args=init_args,\n'
        '        engine_args=engine_args,\n'
        '        peft_args=peft_args,\n'
        '        tinker_args=config.get("tinker_args"),\n'
        '        trainer_args=trainer_args,\n'
        '    )'
    )
    new_return = (
        '    result = InternalModelConfig(\n'
        '        init_args=init_args,\n'
        '        engine_args=engine_args,\n'
        '        peft_args=peft_args,\n'
        '        tinker_args=config.get("tinker_args"),\n'
        '        trainer_args=trainer_args,\n'
        '    )\n'
        '    if config.get("training_device"):\n'
        '        result["training_device"] = config["training_device"]\n'
        '    return result'
    )

    if old_return not in src:
        print("[SKIP] Patch 6: return block not found (already modified?)")
        return False

    patched = src.replace(old_return, new_return, 1)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 6: forward training_device in {path}")
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
    p4 = patch_unsloth_training_device(sp, dry_run=args.check)
    p5 = patch_split_gpu_sleep_wake(sp, dry_run=args.check)
    p6 = patch_get_model_config_training_device(sp, dry_run=args.check)

    if args.check:
        if p1 or p2 or p3 or p4 or p5 or p6:
            print("\nRun without --check to apply patches.")
            sys.exit(1)
        else:
            print("\nAll patches already applied.")
    else:
        if not p1 and not p2 and not p3 and not p4 and not p5 and not p6:
            print("\nNothing to patch.")


if __name__ == "__main__":
    main()
