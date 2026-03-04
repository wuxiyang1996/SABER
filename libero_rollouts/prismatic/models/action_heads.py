"""L1 regression action head for OpenVLA-OFT (prismatic shim)."""

import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)
