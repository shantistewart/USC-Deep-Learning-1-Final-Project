"""PyTorch MLP class.

Call to use a PyTorch MLP model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, input_dim=15, output_dim=2):
        super(_MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim
        self.hidden1 = nn.Linear(input_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = torch.sigmoid(self.output(x))

        return x
