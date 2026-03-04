"""Proprio projector for OpenVLA-OFT (prismatic shim)."""

import torch.nn as nn


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)

    def forward(self, proprio):
        return self.fc2(self.act_fn1(self.fc1(proprio)))
