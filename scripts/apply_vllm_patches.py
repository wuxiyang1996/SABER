#!/usr/bin/env python3
"""
Apply ART 0.5.x ↔ vLLM 0.11.x compatibility patches.

Nine patches:
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
  6. Preserve training_device through get_model_config() in ART's
     dev/get_model_config.py (it rebuilds InternalModelConfig from scratch
     and would drop the key otherwise).
  7. Patch accelerate's prepare_model() to check for ACTUAL bitsandbytes
     quantized layers instead of trusting the is_loaded_in_8bit flag.
     Unsloth unconditionally sets is_loaded_in_8bit=True on ALL models
     (even bf16) to prevent DDP wrapping.  This makes accelerate raise
     ValueError when hf_device_map points to a non-default device (split-
     GPU mode).  Instance-level monkey-patching of for_training() doesn't
     work because Unsloth re-assigns the method on every call.  Patching
     the compiled UnslothGRPOTrainer.py also fails because Unsloth's
     compiler regenerates it on every import.
  8. Relax GRPO reward-equality checks so near-identical rewards still
     produce gradient signal.  ART skips training when all trajectories
     in a group share the exact same reward (set-based equality).  For
     tool-calling agents, structured output suppresses sampling diversity,
     causing all trajectories to produce identical attacks/rewards.
       a) backend.py: use max−min tolerance instead of set uniqueness.
       b) tokenize.py: skip advantage only when |advantage| < 1e-8.
  9. Fix double-encoded tool_call arguments in langchain-core's
     parse_tool_call().  Some OpenAI-compatible servers (vLLM, etc.)
     double-encode the function.arguments field, so json.loads() returns
     a string instead of a dict.  Pydantic v2 then rejects the AIMessage.
     This patch unwraps the extra encoding layer.

Usage:
    python scripts/apply_vllm_patches.py          # auto-detect site-packages
    python scripts/apply_vllm_patches.py --check   # dry-run: report status only
    python scripts/apply_vllm_patches.py --site-packages /path/to/site-packages  # when not in env
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys


def _site_packages() -> str:
    """Return the site-packages (or dist-packages) directory for the active Python."""
    for p in sys.path:
        if p and os.path.isdir(p) and (
            p.endswith("site-packages") or p.endswith("dist-packages")
        ):
            return p
    raise RuntimeError(
        "Could not find site-packages or dist-packages in sys.path. "
        "Run from your conda/venv (e.g. conda activate vast) or pass --site-packages /path/to/site-packages"
    )


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


def patch_accelerate_bnb_check(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 7: fix accelerate's quantization device check.

    accelerate's prepare_model() checks model.is_loaded_in_8bit / is_loaded_in_4bit
    flags to decide whether to enforce device placement rules.  Unsloth sets
    is_loaded_in_8bit=True on ALL models (even bf16) to prevent DDP wrapping,
    causing a false-positive ValueError in split-GPU mode.

    This patch replaces the flag-based check with an actual scan for
    bitsandbytes quantized layers (Linear4bit / Linear8bitLt).

    Why other approaches fail:
    - Monkey-patching model.for_training(): Unsloth re-assigns the method
      via functools.partial on every call (llama.py:3188).
    - Patching UnslothGRPOTrainer.py: Unsloth's compiler detects content
      changes and regenerates the file on every import.
    - Clearing the flag in service.py: for_training() re-sets it before
      accelerator.prepare() runs.
    """
    path = os.path.join(sp, "accelerate", "accelerator.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (accelerate not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    if "_has_bnb_layers" in src:
        print("[OK]   Patch 7: accelerate bnb layer check already present")
        return False

    old_check = (
        '        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(\n'
        '            model, "hf_device_map", False\n'
        "        ):"
    )
    if old_check not in src:
        print("[SKIP] Patch 7: anchor not found in accelerate (version changed?)")
        return False

    if dry_run:
        print("[NEED] Patch 7: replace flag-based quant check with actual bnb layer scan")
        return True

    new_check = (
        "        # Check for ACTUAL bitsandbytes quantized layers, not just flags.\n"
        "        # Unsloth sets is_loaded_in_8bit=True on all models (even bf16) to\n"
        "        # prevent DDP wrapping, which causes false positives here.\n"
        "        def _has_bnb_layers(m):\n"
        "            try:\n"
        "                import bitsandbytes as bnb\n"
        "                return any(isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)) for mod in m.modules())\n"
        "            except (ImportError, AttributeError):\n"
        "                return False\n"
        '        if _has_bnb_layers(model) and getattr(model, "hf_device_map", False):'
    )
    patched = src.replace(old_check, new_check, 1)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 7: replaced flag-based check with bnb layer scan in {path}")
    return True


def patch_grpo_reward_tolerance(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 8: relax GRPO trainability checks so near-identical rewards still train.

    ART skips training when all trajectories in every group have the *exact*
    same reward (set-based equality).  For tool-calling agents the structured
    output format suppresses sampling diversity, so a group of 8 trajectories
    often produces identical attacks → identical rewards → zero gradient
    updates for the entire run.

    Two sub-patches:
      a) art/local/backend.py  — trainability metric uses a tolerance instead
         of exact ``set()`` equality.
      b) art/preprocessing/tokenize.py — per-trajectory advantage skip uses
         ``abs(advantage) < eps`` instead of ``advantage == 0``.
    """
    changed = False

    # --- 8a: backend.py trainability check --------------------------------
    path_a = os.path.join(sp, "art", "local", "backend.py")
    if os.path.isfile(path_a):
        with open(path_a) as f:
            src_a = f.read()

        old_trainable = (
            "        num_groups_trainable = sum(\n"
            "            1\n"
            "            for group in trajectory_groups\n"
            "            if group and len(set(trajectory.reward for trajectory in group)) > 1\n"
            "        )"
        )
        new_trainable = (
            "        num_groups_trainable = sum(\n"
            "            1\n"
            "            for group in trajectory_groups\n"
            "            if group and (max(t.reward for t in group) - min(t.reward for t in group)) > 1e-6\n"
            "        )"
        )

        if "1e-6" in src_a and "max(t.reward" in src_a:
            print("[OK]   Patch 8a: GRPO trainability tolerance already present")
        elif old_trainable in src_a:
            if dry_run:
                print("[NEED] Patch 8a: relax GRPO trainability check in backend.py")
                return True
            patched_a = src_a.replace(old_trainable, new_trainable, 1)
            with open(path_a, "w") as f:
                f.write(patched_a)
            print(f"[DONE] Patch 8a: relaxed GRPO trainability check in {path_a}")
            changed = True
        else:
            print("[SKIP] Patch 8a: trainability anchor not found in backend.py")
    else:
        print(f"[SKIP] Patch 8a: {path_a} not found")

    # --- 8b: tokenize.py advantage skip -----------------------------------
    path_b = os.path.join(sp, "art", "preprocessing", "tokenize.py")
    if os.path.isfile(path_b):
        with open(path_b) as f:
            src_b = f.read()

        old_skip = "            if advantage == 0:\n                continue"
        new_skip = "            if abs(advantage) < 1e-8:\n                continue"

        if "abs(advantage)" in src_b:
            print("[OK]   Patch 8b: advantage tolerance already present")
        elif old_skip in src_b:
            if dry_run:
                print("[NEED] Patch 8b: relax advantage skip in tokenize.py")
                return True
            patched_b = src_b.replace(old_skip, new_skip, 1)
            with open(path_b, "w") as f:
                f.write(patched_b)
            print(f"[DONE] Patch 8b: relaxed advantage skip in {path_b}")
            changed = True
        else:
            print("[SKIP] Patch 8b: advantage skip anchor not found in tokenize.py")
    else:
        print(f"[SKIP] Patch 8b: {path_b} not found")

    if not changed and not dry_run:
        print("[OK]   Patch 8: GRPO reward tolerance already applied")
    return changed


def patch_langchain_double_encoded_tool_args(sp: str, *, dry_run: bool = False) -> bool:
    """Patch 9: fix double-encoded tool_call arguments in langchain-core.

    Some OpenAI-compatible model servers (vLLM, etc.) double-encode the
    function.arguments field in tool calls.  The value arrives as a JSON
    string that, when parsed once by json.loads(), yields another *string*
    instead of a dict.  langchain-core's parse_tool_call() passes this
    string as ``args`` to the ToolCall, and Pydantic v2 validation on
    AIMessage rejects it (``Input should be a valid dictionary``).

    The LoggingLLM wrapper in ART has a post-processing fix for string
    args, but it never executes because the error is raised inside the
    underlying ChatOpenAI.ainvoke() before returning.

    This patch adds a second json.loads() pass when the first parse
    returns a string, and falls back to {} if args is still not a dict.
    """
    path = os.path.join(sp, "langchain_core", "output_parsers", "openai_tools.py")
    if not os.path.isfile(path):
        print(f"[SKIP] {path} not found (langchain-core not installed?)")
        return False

    with open(path) as f:
        src = f.read()

    if "isinstance(function_args, str)" in src:
        print("[OK]   Patch 9: double-encoded tool_call args fix already present")
        return False

    old_block = (
        '    parsed = {\n'
        '        "name": raw_tool_call["function"]["name"] or "",\n'
        '        "args": function_args or {},\n'
        '    }'
    )
    new_block = (
        '    if isinstance(function_args, str):\n'
        '        try:\n'
        '            function_args = json.loads(function_args, strict=strict)\n'
        '        except (JSONDecodeError, TypeError):\n'
        '            pass\n'
        '    parsed = {\n'
        '        "name": raw_tool_call["function"]["name"] or "",\n'
        '        "args": function_args if isinstance(function_args, dict) else {},\n'
        '    }'
    )

    if old_block not in src:
        print("[SKIP] Patch 9: anchor not found in openai_tools.py (already modified or version changed?)")
        return False

    if dry_run:
        print("[NEED] Patch 9: fix double-encoded tool_call arguments in parse_tool_call()")
        return True

    patched = src.replace(old_block, new_block, 1)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[DONE] Patch 9: fixed double-encoded tool_call args in {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Dry-run: report status only")
    parser.add_argument(
        "--site-packages",
        type=str,
        default=None,
        help="Path to site-packages (e.g. from conda env). If not set, auto-detect from sys.path.",
    )
    args = parser.parse_args()

    if args.site_packages:
        sp = os.path.abspath(args.site_packages)
        if not os.path.isdir(sp):
            raise RuntimeError(f"--site-packages path is not a directory: {sp}")
    else:
        sp = _site_packages()
    print(f"site-packages: {sp}\n")

    p1 = patch_async_llm(sp, dry_run=args.check)
    p2 = patch_unsloth_service(sp, dry_run=args.check)
    p3 = patch_tool_parser_import(sp, dry_run=args.check)
    p4 = patch_unsloth_training_device(sp, dry_run=args.check)
    p5 = patch_split_gpu_sleep_wake(sp, dry_run=args.check)
    p6 = patch_get_model_config_training_device(sp, dry_run=args.check)
    p7 = patch_accelerate_bnb_check(sp, dry_run=args.check)
    p8 = patch_grpo_reward_tolerance(sp, dry_run=args.check)
    p9 = patch_langchain_double_encoded_tool_args(sp, dry_run=args.check)

    any_needed = p1 or p2 or p3 or p4 or p5 or p6 or p7 or p8 or p9
    if args.check:
        if any_needed:
            print("\nRun without --check to apply patches.")
            sys.exit(1)
        else:
            print("\nAll patches already applied.")
    else:
        if not any_needed:
            print("\nNothing to patch.")


if __name__ == "__main__":
    main()
