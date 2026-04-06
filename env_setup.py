"""Shared environment setup: GPU resolution and cache directory configuration.

Extracted from train_vla.py and cold_start/collect.py to eliminate duplication.
All functions are stateless utilities — callers manage their own module-level
state (e.g. _orig_gpu_list, CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def setup_cache_dirs(project_root: Optional[str] = None) -> str:
    """Set OPENPI_DATA_HOME, HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, TORCH_HOME.

    All caches go under ``<project_root>/.cache`` so shared clusters don't
    fill ``~/.cache``.

    Parameters
    ----------
    project_root : str, optional
        Root of the project tree.  Defaults to the parent of the directory
        containing this file (i.e. ``agent_attack_framework/..``).

    Returns
    -------
    str
        Resolved absolute path of the cache root directory.
    """
    if project_root is None:
        project_root = os.path.realpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        )
    cache_root = os.path.realpath(os.path.join(project_root, ".cache"))
    os.environ.setdefault("OPENPI_DATA_HOME", cache_root)
    os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_root, "huggingface", "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    try:
        os.makedirs(cache_root, exist_ok=True)
    except OSError:
        pass
    return cache_root


def early_resolve_vla_gpus(vla_gpus_str: Optional[str] = None) -> List[int]:
    """Resolve VLA GPU indices from an explicit string, CLI args, or env vars.

    Resolution order:
      1. *vla_gpus_str* if provided (e.g. from a pre-parsed argparse value).
      2. ``--vla_gpu`` / ``--vla_gpus`` on the command line.
      3. ``VLA_GPUS`` or ``VLA_GPU`` environment variable.
      4. Default ``"0,1,2"``.
    """
    if vla_gpus_str is not None:
        return [int(g.strip()) for g in vla_gpus_str.split(",")]
    raw = None
    for i, tok in enumerate(sys.argv):
        if tok in ("--vla_gpu", "--vla_gpus") and i + 1 < len(sys.argv):
            raw = sys.argv[i + 1]
            break
        if tok.startswith("--vla_gpu=") or tok.startswith("--vla_gpus="):
            raw = tok.split("=", 1)[1]
            break
    if raw is None:
        raw = os.environ.get("VLA_GPUS", os.environ.get("VLA_GPU", "0,1,2"))
    return [int(g.strip()) for g in raw.split(",")]


def logical_to_physical(logical_id: int, physical_ids: List[str]) -> str:
    """Map a logical GPU index to the physical ID from SLURM's visible list.

    Parameters
    ----------
    logical_id : int
        Zero-based logical GPU index (as used in ``--vla_gpus``).
    physical_ids : list[str]
        Physical GPU IDs parsed from the original ``CUDA_VISIBLE_DEVICES``
        (e.g. ``["4", "5", "6", "7"]`` on a SLURM allocation).

    Returns
    -------
    str
        The physical GPU ID string.  Falls back to ``str(logical_id)`` when
        *physical_ids* is empty or the index is out of range.
    """
    if physical_ids and logical_id < len(physical_ids):
        return physical_ids[logical_id]
    return str(logical_id)
