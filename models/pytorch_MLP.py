"""PyTorch MLP class.

Call to use a PyTorch MLP model
"""

import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, input_dim=15, output_dim=2):
        super(_MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim
        self.hidden1 = nn.Linear(input_dim, self.hidden_dim)
        self.hidden2 = nn.Linear(self.hidden_dim, 4)
        self.output = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim=1)

        return x
