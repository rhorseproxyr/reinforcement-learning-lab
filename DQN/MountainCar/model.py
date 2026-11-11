import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        # Initialisation des couches du NN
        self.couche1 = nn.Linear(n_input, 64)
        self.couche2 = nn.Linear(64, n_output)
    def forward(self, x):
        x = self.couche1(x)
        x = F.relu(x)
        return self.couche2(x)