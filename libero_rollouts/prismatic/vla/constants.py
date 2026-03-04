"""Minimal LIBERO constants for OpenVLA-OFT inference (prismatic shim)."""

from enum import Enum

IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2

class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
