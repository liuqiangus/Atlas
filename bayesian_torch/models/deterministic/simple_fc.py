from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SFC(nn.Module):
    def __init__(self, input_dim=14, output_dim=1, activation=F.relu):
        super(SFC, self).__init__()
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        if self.activation is None:
            output = x
        else:
            output = self.activation(x) # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output)
