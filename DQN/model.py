import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialisation des couches du NN
        self.couche1 = nn.Linear(4, 64)
        self.couche2 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.couche1(x)
        x = F.relu(x)
        return self.couche2(x)