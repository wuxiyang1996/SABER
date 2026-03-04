"""Minimal train_utils for OpenVLA-OFT HF model loading (prismatic shim)."""

import torch
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask


def get_next_actions_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask
    return mask
