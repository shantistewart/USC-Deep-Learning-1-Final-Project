"""PyTorch MLP class.

Call to use a PyTorch MLP model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, num_hidden_units, hidden_size, input_dim=15, output_dim=1):
        super(_MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_size
        self.hidden1 = nn.Linear(input_dim, self.hidden_dim)
        self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.num_hidden_units = num_hidden_units

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        for i in range(self.num_hidden_units):
            # reinitialize weights for each new layer
            self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            x = F.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))

        return x
